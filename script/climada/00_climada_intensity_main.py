from pathlib import Path
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
from climada.hazard import TCTracks, TropCyclone, Centroids
import gc
import re
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import os
import rasterio  # type: ignore 
from rasterio import features  # type: ignore
from shapely.geometry import Point, box, mapping  # type: ignore
import zarr  # type: ignore
import argparse
from datetime import datetime, timedelta
import shutil
from rra_tools.parallel import run_parallel  # type: ignore
from scipy.ndimage import binary_dilation  # type: ignore

import logging

logging.getLogger("climada").setLevel(logging.WARNING)


parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--draw_batch", type=str, required=True, help="Draw batch (e.g., '0-9')")
parser.add_argument("--num_cores", type=int, default=1, help="Number of cores to use for parallel processing")

# Parse arguments
args = parser.parse_args()
source_id = args.source_id
variant_label = args.variant_label
experiment_id = args.experiment_id
batch_year = args.batch_year
basin = args.basin
draw_batch = args.draw_batch
num_cores = args.num_cores

# Constants
ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage0_sample_02_25_26")
LOG_DIR = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage0_log_sample_02_25_26/") # TEST
RESOLUTION = 0.1  # degrees
GDF_PATH = Path("/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/shapes_gbd_2021.parquet")

######################################
#        Read in Tracks              #
######################################

def read_custom_tracks_nc(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
) -> Path:

    start_year, end_year = batch_year.split("-")
    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    nc_file = (
        ROOT_PATH
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"tracks_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.nc"
    )

    if not nc_file.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_file}")

    return nc_file


def read_single_storm_from_dataset(ds_all, storm_index: int) -> xr.Dataset:
    """
    Slice a single storm from an open dataset, returning only valid (non-NaN) time steps.
    Optimized to minimize memory and computation for long tracks.
    """
    if storm_index >= ds_all.sizes["n_trk"]:
        raise IndexError(f"Storm index {storm_index} out of range")

    ds_track = ds_all.isel(n_trk=storm_index)

    core_vars = [
        v for v in [
            "lon", "lat", "max_sustained_wind",
            "central_pressure", "environmental_pressure"
        ] if v in ds_track
    ]

    # Start from first and last non-NaN across any variable
    t0, t1 = None, None
    for t in range(ds_track.sizes["time"]):
        if any(not np.isnan(ds_track[var].values[t]) for var in core_vars):
            t0 = t
            break

    for t in reversed(range(ds_track.sizes["time"])):
        if any(not np.isnan(ds_track[var].values[t]) for var in core_vars):
            t1 = t + 1  # slice end exclusive
            break

    if t0 is None or t1 is None or t1 <= t0:
        raise ValueError(f"Storm {storm_index} contains no valid data")

    return ds_track.isel(time=slice(t0, t1)).load()




def normalize_nc_storm_for_climada(ds_track: xr.Dataset) -> xr.Dataset:
    """
    Convert a single-track NC slice into the exact CLIMADA-compatible structure.
    
    Keeps arrays square (padded time steps included), but computes
    start_date and end_date using only valid (non-NaN) core variable time steps.
    """
    # --- Build time coordinate ---
    start_year = int(ds_track["tc_years"].values)
    start_month = int(ds_track["tc_month"].values)
    start_dt = datetime(start_year, start_month, 1)

    # Convert time (seconds) to datetime
    time_seconds = ds_track["time"].values
    time_dt = np.array([start_dt + timedelta(seconds=float(t)) for t in time_seconds])
    n_time = time_dt.size

    # --- Identify valid time steps ---
    core_vars = ["lon", "lat", "max_sustained_wind", "central_pressure", "environmental_pressure"]
    valid_mask = np.all([~np.isnan(ds_track[var].values) for var in core_vars], axis=0)

    # Compute start and end date based only on valid steps
    time_dt_valid = time_dt[valid_mask]
    start_date_iso = time_dt_valid[0].date().isoformat()
    end_date_iso = time_dt_valid[-1].date().isoformat()

    # --- Core variables ---
    lon = ds_track["lon"].values
    lat = ds_track["lat"].values
    vmax = ds_track["max_sustained_wind"].values
    cp = ds_track["central_pressure"].values
    env = ds_track["environmental_pressure"].values

    # --- Basin coordinate ---
    basin = np.repeat(str(ds_track["tc_basins"].values), n_time)

    # Normalize longitude if necessary
    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180

    # --- Time step ---
    dt_hours = float(ds_track["time_step"].values[0])

    # --- Metadata ---
    sid = int(ds_track["n_trk"].values)
    category = int(ds_track["category"].values)

    # --- Build CLIMADA-compatible Dataset ---
    ds = xr.Dataset(
        coords={"time": time_dt},
        data_vars={
            "lon": (("time",), lon),
            "lat": (("time",), lat),
            "max_sustained_wind": (("time",), vmax),
            "central_pressure": (("time",), cp),
            "environmental_pressure": (("time",), env),
            "basin": (("time",), basin),
            "radius_max_wind": (("time",), np.zeros(n_time)),
            "radius_oci": (("time",), np.zeros(n_time)),
            "time_step": (("time",), np.full(n_time, dt_hours)),
        },
        attrs={
            "name": f"storm_{sid:04d}",
            "start_date": start_date_iso,
            "end_date": end_date_iso,
            "storm_basin": basin[0],
            "sid": sid,
            "id_no": sid,
            "category": category,
            "orig_event_flag": True,
            "data_provider": "custom",
            "max_sustained_wind_unit": "kn",
            "central_pressure_unit": "mb",
        },
    )

    return ds


def prepare_track_for_climada(ds_track: xr.Dataset):
    tc_track = normalize_nc_storm_for_climada(ds_track)
    return TCTracks(data=[tc_track])


############################################
#              Helper Functions            #
############################################
    
def normalize_lon(lon: float) -> float:
    """Normalize longitude to [-180, 180] range."""
    lon = ((lon + 180) % 360) - 180
    return lon

def chmod_recursive(path: Path, mode: int = 0o775):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)



#####################################
#    Basin Centroid Functions        #
######################################

def generate_basin_centroids(
    basin: str,
    res: float = 0.1,
    buffer_deg: float = 5.0,
) -> Centroids:
    """
    Generate Centroids for a specific tropical cyclone basin.

    - Uses 0–360 longitude convention (IBTrACS-consistent)
    - Adds a configurable buffer to avoid edge clipping
    - Safely handles storms crossing the 180° meridian
    """

    basin_bounds = {
        'EP': ['180E', '0N', '290E', '60N'],
        'NA': ['260E', '0N', '360E', '60N'],
        'NI': ['30E',  '0N', '100E', '50N'],
        'SI': ['20E',  '45S', '100E', '0S'],
        'AU': ['100E', '45S', '180E', '0S'],
        'SP': ['180E', '45S', '250E', '0S'],
        'WP': ['100E', '0N', '180E', '60N'],
    }


    if basin not in basin_bounds:
        raise ValueError(
            f"Basin '{basin}' not recognized. "
            f"Available basins: {list(basin_bounds.keys())}"
        )

    def parse_coord(coord_str: str) -> float:
        """
        Convert coordinate string (e.g. '250E', '45S') to float degrees.

        Longitude stays in 0–360 space.
        Latitude stays in [-90, 90].
        """
        match = re.match(r"([0-9\.]+)([ENWS])", coord_str)
        if not match:
            raise ValueError(f"Invalid coordinate string: {coord_str}")

        val, hemi = match.groups()
        val = float(val)

        if hemi == 'S':
            val = -val
        elif hemi == 'W':
            val = 360.0 - val  # explicit 0–360 handling

        return val

    lon_min, lat_min, lon_max, lat_max = [
        parse_coord(c) for c in basin_bounds[basin]
    ]

    # Apply buffer
    lon_min -= buffer_deg
    lon_max += buffer_deg
    lat_min -= buffer_deg
    lat_max += buffer_deg

    # Expand upper bounds to include last grid cell
    lon_max += res
    lat_max += res

    # Create centroids
    centroids = Centroids.from_pnt_bounds(
        (lon_min, lat_min, lon_max, lat_max),
        res=res,
    )

    return centroids


######################################
#    Hazard Generation Functions     #
######################################

def generate_hazard_per_track(tc_tracks: TCTracks, centroids: Centroids) -> TropCyclone:
    """
    Generate CLIMADA TropCyclone hazard object from TCTracks and Centroids.
    """

    haz = TropCyclone.from_tracks(tc_tracks, centroids=centroids, store_windfields=True)

    return haz

######################################
#    Wind Speed Generation Functions #
######################################

def generate_speed_per_storm(
    haz: TropCyclone,
    centroids: Centroids,
    tc_tracks: TCTracks,
    buffer_deg: float = 5.0,
) -> xr.DataArray:
    """
    Generate per-storm wind speed DataArray, cropped to storm footprint,
    normalized longitude (-180..180), and zeros outside storm footprint removed.
    """
    # --- Coordinates ---
    lon = np.unique(centroids.coord[:, 1])
    lat = np.unique(centroids.coord[:, 0])
    lat_desc = np.sort(lat)[::-1]  # descending

    n_lat = len(lat)
    n_lon = len(lon)

    event = tc_tracks.data[0]
    storm_name = event.name
    storm_id = event.sid
    storm_start_date = event.start_date
    storm_end_date = event.end_date
    storm_basin = getattr(event, "storm_basin", None)
    storm_category = getattr(event, "category", None)
    times = event.time
    wf = haz.windfields[0].toarray()  # shape (time, n_centroids, 2)
    n_time = len(times)

    # --- Reshape windfield ---
    try:
        wf_reshaped = wf.reshape(n_time, n_lat, n_lon, 2)
    except ValueError:
        print(f"⚠️ Skipping storm {storm_name}: shape mismatch")
        return xr.DataArray()  # return empty DA

    # --- Create DataArray ---
    da = xr.DataArray(
        wf_reshaped,
        coords={"time": times, "lat": lat_desc, "lon": lon, "dir": ["u", "v"]},
        dims=["time", "lat", "lon", "dir"],
        name=f"{storm_name}_windfields"
    )

    # --- Compute wind speed ---
    da_speed = np.sqrt(da.isel(dir=0)**2 + da.isel(dir=1)**2)
    da_speed.attrs.update({
        "description": f"Storm {storm_name} wind speed",
        "units": "m/s",
        "storm_name": storm_name,
        "storm_id": storm_id,
        "start_date": storm_start_date,
        "end_date": storm_end_date,
        "basin": storm_basin,
        "category": storm_category,
    })

    # --- Free memory ---
    del wf, wf_reshaped, da

    return da_speed


######################################
#    Yearly Exposure Functions       #
######################################

def compute_yearly_exposure_per_storm(
    storm_da: xr.DataArray,
    wind_threshold: float = 17.0,
) -> xr.DataArray:
    """
    Compute per-storm, per-year exposure hours at the pixel level.

    Exposure is defined as the number of timesteps where wind speed
    is >= wind_threshold. Each timestep is assumed to represent 1 hour.

    Longitude is normalized (-180..180) at the very end.
    """

    if "time" not in storm_da.coords:
        raise ValueError(f"Storm {storm_da.name} missing 'time' coordinate")

    # --------------------------------------------------------
    # 1. Threshold → exposure mask (1 hour per timestep)
    # --------------------------------------------------------
    exposure = xr.where(storm_da > wind_threshold, 1.0, 0.0)

    # --------------------------------------------------------
    # 2. Group by year
    # --------------------------------------------------------
    time_index = pd.DatetimeIndex(storm_da["time"].values)
    year_groups = time_index.to_period("Y").to_timestamp()

    group_da = xr.DataArray(
        year_groups,
        dims="time",
        coords={"time": exposure.time},
        name="year",
    )

    yearly_exposure = exposure.groupby(group_da).sum(dim="time")

    # normalize dimension name
    if "year" in yearly_exposure.dims:
        yearly_exposure = yearly_exposure.rename({"year": "time"})

    yearly_exposure = yearly_exposure.assign_coords(
        time=np.array(yearly_exposure.time.values, dtype="datetime64[ns]")
    )

    yearly_exposure = yearly_exposure.astype("float32")

    # --------------------------------------------------------
    # 3. Metadata
    # --------------------------------------------------------
    yearly_exposure.name = "exposure_hours"
    yearly_exposure.attrs.update({
        "storm_name": storm_da.attrs.get("storm_name"),
        "storm_id": storm_da.attrs.get("storm_id"),
        "start_date": storm_da.attrs.get("start_date"),
        "end_date": storm_da.attrs.get("end_date"),
        "basin": storm_da.attrs.get("basin"),
        "category": storm_da.attrs.get("category"),
        "description": (
            f"Per-storm yearly exposure hours per pixel "
            f"where wind speed > {wind_threshold} m/s"
        ),
        "definition": (
            "Exposure hours are computed as the number of timesteps "
            "with wind speed above the threshold. Each timestep "
            "is assumed to represent one hour."
        ),
        "units": "hours",
        "aggregation": "yearly",
        "wind_threshold_m_s": wind_threshold,
    })

    # Remove timestep-specific attrs if present
    yearly_exposure.attrs.pop("time_step", None)

    # After yearly aggregation
    if yearly_exposure.sizes.get("time", 0) == 1:
        yearly_exposure = yearly_exposure.isel(time=0, drop=True)

    # ---- Check if storm exposure is all zeros ----
    data_max = float(yearly_exposure.max().values)

    if data_max == 0:
        # Return full-zero array with same coords and attrs
        empty_da = xr.DataArray(
            np.zeros((yearly_exposure.sizes["lat"], yearly_exposure.sizes["lon"]), dtype=float),
            coords={"lat": yearly_exposure["lat"], "lon": yearly_exposure["lon"]},
            dims=["lat", "lon"],
            name="exposure_hours",
        )
        empty_da.attrs.update(yearly_exposure.attrs)
        # Only normalize longitude, skip cropping
        lon_180 = ((empty_da["lon"].values + 180) % 360) - 180
        empty_da = empty_da.assign_coords(lon=lon_180).sortby("lon")
        return empty_da

    # ---- For valid storms ----
    # Remove empty space (zeros) outside storm footprint
    yearly_exposure = yearly_exposure.where(yearly_exposure > 0)
    yearly_exposure = yearly_exposure.dropna(dim="lat", how="all")
    yearly_exposure = yearly_exposure.dropna(dim="lon", how="all")

    BUFFER = 0.1

    # --------------------------------------------------
    # Expand single-pixel footprint by creating new grid
    # --------------------------------------------------
    if yearly_exposure.lat.size == 1:
        center = float(yearly_exposure.lat.values[0])
        new_lat = np.array([center - BUFFER, center, center + BUFFER])
        yearly_exposure = yearly_exposure.reindex(lat=new_lat, fill_value=0)

    if yearly_exposure.lon.size == 1:
        center = float(yearly_exposure.lon.values[0])
        new_lon = np.array([center - BUFFER, center, center + BUFFER])
        yearly_exposure = yearly_exposure.reindex(lon=new_lon, fill_value=0)

    # ---- Normalize longitude to -180..180 ----
    lon_vals = yearly_exposure["lon"].values
    lon_180 = ((lon_vals + 180) % 360) - 180
    yearly_exposure = yearly_exposure.assign_coords(lon=lon_180)
    yearly_exposure = yearly_exposure.sortby("lon")

    # ---- Fill remaining NaNs with zeros ----
    yearly_exposure = yearly_exposure.fillna(0)
    
    return yearly_exposure


######################################
#    Per Storm Intensity Functions   #
######################################

def generate_intensity_per_storm(
    haz: TropCyclone,
    centroids: Centroids,
    tc_tracks: TCTracks,
) -> xr.DataArray:
    """
    Generate per-storm, per-pixel intensity using CLIMADA haz.intensity.

    Intensity is defined as the maximum wind speed experienced at each pixel
    during the storm lifetime.

    Returns
    -------
    xr.DataArray
        DataArray with dims ('lat', 'lon') representing the maximum wind speed per pixel for all storms.
    """
    lon = np.unique(centroids.coord[:, 1])

    lat = np.unique(centroids.coord[:, 0])
    lat = np.sort(lat)
    lat_desc = lat[::-1]   # descending, north → south

    n_lat = len(lat)
    n_lon = len(lon)

    event = tc_tracks.data[0]

    storm_name = event.name
    storm_basin = event.basin
    storm_category = event.category
    storm_id = event.sid
    start_date = event.start_date
    end_date = event.end_date


    try:
        # shape: (n_centroids,)
        intensity_flat = haz.intensity.toarray()[0, :]
    except Exception as e:
        print(f"⚠️ Could not read intensity for {storm_name}: {e}")

    if intensity_flat.size != n_lat * n_lon:
        print(
            f"⚠️ Skipping {storm_name}: grid mismatch "
            f"{intensity_flat.size} vs {n_lat * n_lon}"
        )
        raise ValueError(f"Grid size mismatch for {storm_name}")

    # reshape to 2D grid
    intensity_2d = intensity_flat.reshape(n_lat, n_lon)

    da = xr.DataArray(
        intensity_2d,
        coords={"lat": lat_desc, "lon": lon},
        dims=["lat", "lon"],
        name=f"{storm_name}_intensity",
    )

    da.attrs.update({
        "description": "Per-storm pixel-level maximum wind speed",
        "units": "m/s",
        "storm_name": storm_name,
        "storm_id": storm_id,
        "start_date": start_date,
        "end_date": end_date,
        "basin": storm_basin,
        "category": storm_category,
        "definition": (
            "Maximum wind speed experienced at each pixel "
            "during the storm lifetime")
        })

    data_max = float(da.max().values)

    if data_max == 0:
        # Return full zero array with same coords and attributes
        empty_da = xr.DataArray(
            np.zeros((n_lat, n_lon), dtype=float),
            coords={"lat": lat_desc, "lon": lon},
            dims=["lat", "lon"],
            name=f"{storm_name}_intensity",
        )
        empty_da.attrs.update(da.attrs)
        # Only normalize longitude, skip dropping zeros
        lon_180 = ((lon + 180) % 360) - 180
        empty_da = empty_da.assign_coords(lon=lon_180).sortby("lon")
        return empty_da

    BUFFER = 0.1

    # --------------------------------------------------
    # Expand single-pixel footprint by creating new grid
    # --------------------------------------------------
    if da.lat.size == 1:
        center = float(da.lat.values[0])
        new_lat = np.array([center - BUFFER, center, center + BUFFER])
        da = da.reindex(lat=new_lat, fill_value=0)

    if da.lon.size == 1:
        center = float(da.lon.values[0])
        new_lon = np.array([center - BUFFER, center, center + BUFFER])
        da = da.reindex(lon=new_lon, fill_value=0)

    # ---- Normalize longitude to -180..180 ----
    lon_vals = da["lon"].values.astype("float32")
    lon_180 = ((lon_vals + 180) % 360) - 180
    da = da.assign_coords(lon=lon_180)
    da = da.sortby("lon")

    return da

######################################
#       Track Storm Duration         #
######################################
def track_storm_duration(
    storm_da: xr.DataArray,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
) -> pd.DataFrame:
    """
    Track storm duration in days for each storm.

    Parameters
    ----------
    storm_intensity : xr.DataArray
        DataArray of storm wind speed with dims ('time', 'lat', 'lon').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - storm_name
        - storm_id
        - start_date
        - end_date
        - duration_days
    """


    storm_name = storm_da.attrs.get("storm_name")
    storm_id = storm_da.attrs.get("storm_id")
    start_date = storm_da.attrs.get("start_date")
    end_date = storm_da.attrs.get("end_date")

    # calculate duration in days
    start_dt = datetime.fromisoformat(str(start_date))
    end_dt = datetime.fromisoformat(str(end_date))
    duration_days = (end_dt - start_dt).days + 1  # inclusive

    rows = []
    # if duration is 30 days or more, log a warning
    if duration_days >= 30:
        rows.append({
            "storm_name": storm_name,
            "storm_id": storm_id,
            "start_date": start_date,
            "end_date": end_date,
            "duration_days": duration_days,
            "warning": "Duration exceeds 30 days - check for potential issues",
        })

    df_duration = pd.DataFrame(rows)

    # save log
    if not df_duration.empty:
        log_path = LOG_DIR / f"storm_duration_{source_id}_{variant_label}_{experiment_id}_{batch_year}_{basin}_draw{draw}.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not log_path.exists()
        df_duration.to_csv(
            log_path,
            mode="a",
            header=write_header,
            index=False,
        )


#############################
#       Helper Functions    #
#############################

def get_storm_indices(ds_all: xr.Dataset) -> list[int]:
    """
    Return a list of valid storm indices from an open multi-storm dataset
    for CLIMADA processing. No particular order is enforced.

    Parameters
    ----------
    ds_all : xr.Dataset
        Multi-storm dataset containing 'n_trk' dimension.

    Returns
    -------
    list[int]
        List of valid storm indices.
    """
    n_storms = ds_all.sizes["n_trk"]
    valid_indices = []

    core_vars = ["lon", "lat", "max_sustained_wind", "central_pressure", "environmental_pressure"]

    for storm_index in range(n_storms):
        ds_track = ds_all.isel(n_trk=storm_index)

        # check if there are any valid timesteps across core variables
        valid_mask = np.all([~np.isnan(ds_track[var].values) for var in core_vars], axis=0)
        if valid_mask.any():
            valid_indices.append(storm_index)

    return valid_indices




def sanitize_attrs(attrs: dict) -> dict:
    """
    Recursively sanitize a dictionary of attributes to be Zarr v3 compatible.
    Converts all non-JSON-serializable types to native types or strings.
    """
    safe = {}
    for k, v in attrs.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        elif isinstance(v, (np.integer, np.floating)):
            safe[k] = v.item()
        elif isinstance(v, (np.bool_)):
            safe[k] = bool(v)
        elif isinstance(v, np.datetime64):
            safe[k] = str(v)
        elif isinstance(v, (list, tuple)):
            safe[k] = [
                x.item() if isinstance(x, (np.integer, np.floating)) else
                bool(x) if isinstance(x, np.bool_) else
                str(x)
                for x in v
            ]
        elif isinstance(v, dict):
            safe[k] = sanitize_attrs(v)
        else:
            # last-resort stringify for unknown types
            safe[k] = str(v)
    return safe


#########################################
#     Check Landfall Functions          #
#########################################

def load_land_polygons_for_storm(storm_da, shapefile_gdf_path: str, buffer: float = 5.0) -> gpd.GeoDataFrame:
    """
    Load land polygons that intersect the bounding box of the storm intensity.

    Parameters
    ----------
    storm_da : xr.DataArray
        Storm intensity DataArray (lat/lon) already normalized to -180..180
        and cropped around the storm footprint.
    shapefile_gdf_path : str
        Path to the land polygons parquet file (WGS84, -180..180)
    buffer : float
        Degrees to expand around storm bounding box.

    Returns
    -------
    gpd.GeoDataFrame
        Land polygons intersecting the storm + buffer.
    """

    # ---- Compute bounding box around storm + buffer ----
    min_lon = float(storm_da["lon"].min()) - buffer
    max_lon = float(storm_da["lon"].max()) + buffer
    min_lat = float(storm_da["lat"].min()) - buffer
    max_lat = float(storm_da["lat"].max()) + buffer

    # ---- Load shapefile (no initial bbox filtering) ----
    gdf = gpd.read_parquet(shapefile_gdf_path)

    # ---- Subset polygons that intersect the storm bounding box ----
    storms_bbox_gdf = gpd.GeoDataFrame(
        geometry=[box(min_lon, min_lat, max_lon, max_lat)],
        crs=gdf.crs
    )    

    gdf_subset = gdf[gdf.intersects(storms_bbox_gdf.iloc[0].geometry)].copy()

    # clip to exact storm bbox for efficiency in later processing
    gdf_subset["geometry"] = gdf_subset.geometry.intersection(storms_bbox_gdf.iloc[0].geometry)

    return gdf_subset


def check_storm_landfall(
    storm_intensity: xr.DataArray,
    land_gdf: gpd.GeoDataFrame,
    buffer_km: float = 2,
    wind_threshold: float = 17.0,
) -> bool:
    """
    True only if storm has winds >= threshold AND those winds intersect land.
    """

    # --- 0. Skip globally weak storms ---
    max_intensity = float(storm_intensity.max().values)
    if max_intensity < wind_threshold:
        return False

    # --- Ensure grid is valid ---
    if storm_intensity.lat.size < 2 or storm_intensity.lon.size < 2:
        # too small to rasterize safely
        return False

    # --- 1. Create transform ---
    lons = storm_intensity["lon"].values
    lats = storm_intensity["lat"].values

    res_lon = np.abs(lons[1] - lons[0])
    res_lat = np.abs(lats[1] - lats[0])

    transform = rasterio.transform.from_origin(
        west=lons.min() - res_lon / 2,
        north=lats.max() + res_lat / 2,
        xsize=res_lon,
        ysize=res_lat
    )

    out_shape = (len(lats), len(lons))

    # --- 2. Rasterize land ---
    land_raster = features.rasterize(
        ((geom, 1) for geom in land_gdf.geometry),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # --- 3. Buffer land ---
    if buffer_km > 0:
        avg_lat = float(lats.mean())
        meters_per_deg_lat = 111_000
        meters_per_deg_lon = 111_000 * np.cos(np.deg2rad(avg_lat))

        pixel_buffer_x = int(np.ceil(buffer_km * 1000 / (res_lon * meters_per_deg_lon)))
        pixel_buffer_y = int(np.ceil(buffer_km * 1000 / (res_lat * meters_per_deg_lat)))
        iterations = max(pixel_buffer_x, pixel_buffer_y)

        land_raster = binary_dilation(land_raster, iterations=iterations)

    # --- 4. STRONG wind overlap with land ---
    strong_wind_mask = storm_intensity.values > wind_threshold
    land_mask = land_raster.astype(bool)

    makes_landfall = np.any(strong_wind_mask & land_mask)

    return makes_landfall



def mask_to_land(
    data: xr.DataArray,
    land_gdf: gpd.GeoDataFrame,
    buffer_km: float = 1,
) -> xr.DataArray:
    """
    Mask an xarray DataArray to land pixels only.
    Ocean pixels are set to NaN.

    Parameters
    ----------
    data : xr.DataArray
        Must have 'lat' and 'lon' coordinates.
    land_gdf : gpd.GeoDataFrame
        Land polygons already clipped to region of interest.
    buffer_km : float
        Optional raster-based buffer distance.

    Returns
    -------
    xr.DataArray
        Land-masked DataArray.
    """

    lons = data["lon"].values
    lats = data["lat"].values

    res_lon = np.abs(lons[1] - lons[0])
    res_lat = np.abs(lats[1] - lats[0])

    transform = rasterio.transform.from_origin(
        west=lons.min() - res_lon / 2,
        north=lats.max() + res_lat / 2,
        xsize=res_lon,
        ysize=res_lat
    )

    out_shape = (len(lats), len(lons))

    # --- Rasterize land ---
    land_raster = features.rasterize(
        ((geom, 1) for geom in land_gdf.geometry),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    # --- Optional raster buffer ---
    if buffer_km > 0:
        avg_lat = float(lats.mean())
        meters_per_deg_lat = 111_000
        meters_per_deg_lon = 111_000 * np.cos(np.deg2rad(avg_lat))

        pixel_buffer_x = int(np.ceil(buffer_km * 1000 / (res_lon * meters_per_deg_lon)))
        pixel_buffer_y = int(np.ceil(buffer_km * 1000 / (res_lat * meters_per_deg_lat)))
        iterations = max(pixel_buffer_x, pixel_buffer_y)

        land_raster = binary_dilation(land_raster, iterations=iterations)

    # --- Apply mask ---
    masked = data.where(land_raster)

    return masked

#######################################
#    Save Per Storm Functions         #
#######################################

def save_single_storm_intensity(
    da: xr.DataArray,
    storm_index: int,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    save_root: Path = SAVE_ROOT,
):
    """
    Save a single storm's intensity DataArray to a draw-level Zarr store.

    Parameters
    ----------
    da : xr.DataArray
        Single storm intensity.
    storm_index : int
        Index of the storm (used for Zarr group key).
    source_id, variant_label, experiment_id, batch_year, basin, draw : str/int
        Metadata for path construction.
    save_root : Path, optional
        Root path for saving.
    """
    save_root.mkdir(parents=True, exist_ok=True)

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    draw_store = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / "intensity"
        / f"intensity_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
    )

    draw_store.parent.mkdir(parents=True, exist_ok=True)

    # Use storm index for Zarr group key
    storm_key = f"storm_{storm_index:04d}"  # 4 digits to match original 
    
    # --- check if Zarr store exists and if this storm is already written ---
    if draw_store.exists():
        z = zarr.open(draw_store, mode="a")
        if storm_key in z:
            print(f"⚠️ Storm {storm_index} already exists in {draw_store}, skipping.")
            return
        
    # Defensive copy & cast
    da = da.copy()
    da.name = "intensity"
    if isinstance(da.attrs.get("basin"), xr.DataArray):
        da.attrs["basin"] = str(da.attrs["basin"].values[0])
    if da.dtype != "float32":
        da = da.astype("float32")

    # Chunk the DataArray first with dimension names
    da = da.chunk({"lat": 64, "lon": 64})
    ds = da.to_dataset()
    ds.attrs.update(sanitize_attrs(da.attrs))


    # Then use simpler encoding without chunks specification
    encoding = {
        "intensity": {
            "compressors": [
                {
                    "name": "blosc",
                    "configuration": {
                        "cname": "zstd",
                        "clevel": 9,
                        "shuffle": "bitshuffle",
                    },
                }
            ],
            "dtype": "float32",
            "fill_value": np.nan,
            # Remove chunks from encoding since we chunked the DataArray above
        }
    }

    # Always append to Zarr store; create if it doesn't exist
    ds.to_zarr(
        draw_store,
        group=storm_key,
        mode="a",
        encoding=encoding,
        zarr_format=3,
        consolidated=False,
    )
    # chmod_recursive(draw_store, mode=0o775)


def save_single_storm_exposure(
    da: xr.DataArray,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    storm_index: int,
    save_root: Path = SAVE_ROOT,
):
    """
    Save a single storm's exposure_hours DataArray to a Zarr store.

    If the Zarr store doesn't exist, it will be created automatically.
    """
    save_root.mkdir(parents=True, exist_ok=True)

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    draw_store = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / "exposure_hours"
        / f"exposure_hours_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
    )

    draw_store.parent.mkdir(parents=True, exist_ok=True)

    storm_key = f"storm_{storm_index:04d}"  # 4 digits to match original 

    # --- check if Zarr store exists and if this storm is already written ---
    if draw_store.exists():
        z = zarr.open(draw_store, mode="a")
        if storm_key in z:
            print(f"⚠️ Storm {storm_index} already exists in {draw_store}, skipping.")
            return
        
    # Defensive copy
    da = da.copy()
    da.name = "exposure_hours"

    # Ensure float32
    if da.dtype != "float32":
        da = da.astype("float32")

    # Chunk the DataArray first with dimension names
    da = da.chunk({"lat": 64, "lon": 64})
    ds = da.to_dataset()
    ds.attrs.update(sanitize_attrs(da.attrs))

    # Then use simpler encoding without chunks specification
    encoding = {
        "exposure_hours": {
            "compressors": [
                {
                    "name": "blosc",
                    "configuration": {
                        "cname": "zstd",
                        "clevel": 9,
                        "shuffle": "bitshuffle",
                    },
                }
            ],
            "dtype": "float32",
            "fill_value": np.nan,
            # Remove chunks from encoding since we chunked the DataArray above
        }
    }

    # Always append to Zarr store; create if it doesn't exist
    ds.to_zarr(
        draw_store,
        group=storm_key,
        mode="a",
        encoding=encoding,
        zarr_format=3,
        consolidated=False,
    )
    # chmod_recursive(draw_store, mode=0o775)




#################################
#     Check Existing Files      #
#################################
def check_and_cleanup_zarr_store(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    save_root: Path = SAVE_ROOT,
) -> None:
    """
    Check the top-level Zarr store for a metric. If it exists and contains .partial files,
    delete the entire store to allow a clean rerun.

    Returns True if the store exists and is complete (no partial files), False otherwise.
    """
    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    metrics = ["intensity", "exposure_hours", "days_impact"]

    for metric in metrics:
        draw_store = (
            save_root
            / source_id
            / variant_label
            / experiment_id
            / batch_year
            / basin
            / metric
            / f"{metric}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
        )

        if not draw_store.exists():
            return 

        # If any partial files exist in the top-level Zarr store, delete it completely
        if any(draw_store.glob("*.partial")):
            print(f"⚠️ Found .partial file in {draw_store}. Deleting the entire store for a clean rerun...")
            shutil.rmtree(draw_store, ignore_errors=True)
            return 

    return 


def check_existing_storm_in_zarr(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    storm_index: int,
    save_root: Path = SAVE_ROOT,
) -> bool:

    metrics = ["intensity", "exposure_hours", "days_impact"]

    expected_arrays = {
        "intensity": {"intensity", "lat", "lon"},
        "exposure_hours": {"exposure_hours", "lat", "lon"},
        "days_impact": {"days_impact", "lon", "time"},
    }

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")
    storm_key = f"storm_{storm_index:04d}"

    for metric in metrics:

        draw_store = (
            save_root
            / source_id
            / variant_label
            / experiment_id
            / batch_year
            / basin
            / metric
            / f"{metric}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
        )

        if not draw_store.exists():
            return False

        storm_path = draw_store / storm_key
        if not storm_path.exists():
            return False

        # 🔥 delete partial storms
        if any(storm_path.rglob("*.partial")):
            shutil.rmtree(storm_path, ignore_errors=True)
            print(f"⚠️ Deleted corrupted {storm_key} in {metric}")
            return False

        # 🔍 validate structure
        try:
            g = zarr.open_group(storm_path, mode="r")
            arrays = set(g.array_keys())

            if not expected_arrays[metric].issubset(arrays):
                print(f"⚠️ Invalid structure for {storm_key} in {metric}")
                shutil.rmtree(storm_path, ignore_errors=True)
                return False

        except Exception as e:
            print(f"⚠️ Error reading {storm_key} in {metric}: {e}")
            shutil.rmtree(storm_path, ignore_errors=True)
            return False

    return True


from datetime import datetime
import json

def write_draw_completion_marker(
    log_root: Path,
    source_id,
    variant_label,
    experiment_id,
    batch_year,
    basin,
    draw,
):
    log_root = log_root / "draw_completion_markers"
    marker_dir = (
        log_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
    )
    marker_dir.mkdir(parents=True, exist_ok=True)

    marker_path = marker_dir / f"draw_{draw:04d}.json"

    payload = {
        "source_id": source_id,
        "variant_label": variant_label,
        "experiment_id": experiment_id,
        "batch_year": batch_year,
        "basin": basin,
        "draw": draw,
        "completed_utc": datetime.utcnow().isoformat(),
    }

    tmp_path = marker_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(marker_path)

def is_draw_completed(
    log_root: Path,
    source_id,
    variant_label,
    experiment_id,
    batch_year,
    basin,
    draw,
):
    log_root = log_root / "draw_completion_markers"

    marker_path = (
        log_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"draw_{draw:04d}.json"
    )
    return marker_path.exists()
############################################
#              Main                        #
############################################
        
def process_single_storm(i, storm_index, ds_all, storm_indices, save_root,
    source_id, variant_label, experiment_id, batch_year, basin, draw, centroids) -> None:

    """
    Process a single storm: compute intensity, speed, exposure, and days impact.
    This function is intended for parallel execution over storm indices.

    Parameters
    ----------
    storm_info : tuple
        (storm_index, track_store, storm_indices, centroids, save_root,
         source_id, variant_label, experiment_id, batch_year, basin, draw)
    """

    # Check if storm has already been processed (by checking all three stores)
    if check_existing_storm_in_zarr(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        storm_index=storm_index,
        save_root=save_root,
    ):
        print(f"⚠️ Storm {storm_index} already processed in all metrics, skipping.")
        return
    
    # --- Load storm directly from open dataset ---
    ds_track = read_single_storm_from_dataset(ds_all, storm_index)
    tc_tracks = prepare_track_for_climada(ds_track)

    haz = generate_hazard_per_track(tc_tracks, centroids)

    storm_intensity = generate_intensity_per_storm(haz, centroids, tc_tracks)

    # check landfall
    land_gdf = load_land_polygons_for_storm(storm_intensity, GDF_PATH)
    makes_landfall = check_storm_landfall(storm_intensity, land_gdf, 25) 

    # add small buffer to landfall polygons
    # mask over ocean points as nans - apply mask to intensity, exposure, and days impact to ensure consistency


    if not makes_landfall:
        print(f"⚠️ Storm {storm_index} does not make landfall, skipping exposure and days impact.")
        return

    # track storm duration for logging
    track_storm_duration(
        storm_intensity,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    )

    # mask to land
    storm_intensity = mask_to_land(storm_intensity, land_gdf, buffer_km=25)

    # save storm intensity
    save_single_storm_intensity(
        storm_intensity,
        storm_index=storm_index,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        save_root=save_root,
    )
    del storm_intensity

    storm_speed = generate_speed_per_storm(haz, centroids, tc_tracks)
    storm_exposure = compute_yearly_exposure_per_storm(
        storm_speed,
        wind_threshold=17.0,
    )

    # mask to land
    storm_exposure = mask_to_land(storm_exposure, land_gdf, buffer_km=25)

    # save exposure
    save_single_storm_exposure(
        storm_exposure,
        storm_index=storm_index,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        save_root=save_root,
    )
    del storm_exposure


    del storm_speed
    del tc_tracks, haz
    gc.collect()

def process_single_draw(draw_info):
    (
        source_id,
        variant_label,
        experiment_id,
        batch_year,
        basin,
        draw,
        save_root,
    ) = draw_info

    print(f"▶ Processing draw {draw} | {source_id} {variant_label} {experiment_id} {batch_year} {basin}")

    # check if draw is already completed
    if is_draw_completed(
        log_root=LOG_DIR,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    ):
        print(f"⚠️ Draw {draw} already marked as completed, skipping.")
        return

    # check and cleanup any existing stores with partial files before processing
    check_and_cleanup_zarr_store(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        save_root=save_root,
    )

    nc_file = read_custom_tracks_nc(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    )

    # 🔥 Open once for entire draw
    with xr.open_dataset(nc_file) as ds_all:

        storm_indices = get_storm_indices(ds_all)

        centroids = generate_basin_centroids(basin, res=RESOLUTION)

        for i, storm_index in enumerate(storm_indices):
            process_single_storm(
                i,
                storm_index,
                ds_all,
                storm_indices,
                save_root,
                source_id,
                variant_label,
                experiment_id,
                batch_year,
                basin,
                draw,
                centroids,
            )


    # finalize permissions once per draw
    draw_store = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
    )

    metrics = ["intensity", "exposure_hours"]

    for metric in metrics:
        metric_store = draw_store / metric
        draw_text = "" if draw == 0 else f"_e{draw - 1}"
        start_year, end_year = batch_year.split("-")

        final_zarr_path = (
            metric_store
            / f"{metric}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}_final.zarr"
        )

        try:
            chmod_recursive(final_zarr_path, mode=0o775)
        except Exception as e:
            print(f"⚠️ Could not set permissions for {final_zarr_path}: {e}")

    # log completion of draw as text file
    write_draw_completion_marker(
        log_root=LOG_DIR,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    )

    print(f"✅ Completed draw {draw}")

def main(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw_batch: str,
    num_cores: int = 4,
    save_root: Path = SAVE_ROOT,
):
    """
    Parallelize over draws.
    Storms are processed sequentially within each draw.
    """
    start_draw, end_draw = map(int, draw_batch.split("-"))
    draws = list(range(start_draw, end_draw + 1))

    draw_args = [
        (
            source_id,
            variant_label,
            experiment_id,
            batch_year,
            basin,
            draw,
            save_root,
        )
        for draw in draws
    ]

    run_parallel(
        runner=process_single_draw,
        arg_list=draw_args,
        num_cores=num_cores,
    )

    print("🎉 All draws completed.")



main(
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
    batch_year=batch_year,
    basin=basin,
    draw_batch=draw_batch,
    num_cores=num_cores,
)

"""
103 total source_id, variant_label, experiment_id, batch_year combination parameters
7 total basins
100 draws per combination
In total we have 103 * 7 * 100 = 72,100  unique runs to process
We use multithreading / multiprocessing to parallelize each task over thier given storms


Single draw with 8 cores takes approximately 1-15 minutes and 15-25GB of memory



"""