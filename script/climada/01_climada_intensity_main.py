"""
Stage 1: CLIMADA tropical cyclone intensity, exposure, and per-location landfall.
"""

from pathlib import Path
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
from climada.hazard import TCTracks, TropCyclone, Centroids
import gc
import re
import json
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import os
import zarr  # type: ignore
import argparse
import traceback
from datetime import datetime, timezone
import shutil
from rra_tools.parallel import run_parallel  # type: ignore
import rasterra as rt  # type: ignore
from shapely.geometry import box  # type: ignore
from affine import Affine  # type: ignore
from rasterio.errors import WindowError

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
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1_v2/")
LOG_DIR = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1_v2_log/")
RESOLUTION = 0.1  # degrees
GDF_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84.parquet")
SHP_ROOT_NORMALIZED = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29')

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
    Slice a single storm from an open dataset, returning only valid (non-NaN)
    time steps trimmed to the first/last step where ANY core variable is finite.
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

    # Per-timestep mask: True if ANY core variable is finite at that step.
    any_finite = np.any(
        np.stack([~np.isnan(ds_track[v].values) for v in core_vars], axis=0),
        axis=0,
    )
    valid_idx = np.flatnonzero(any_finite)
    if valid_idx.size == 0:
        raise ValueError(f"Storm {storm_index} contains no valid data")

    t0 = int(valid_idx[0])
    t1 = int(valid_idx[-1]) + 1  # slice end exclusive
    return ds_track.isel(time=slice(t0, t1)).load()




def normalize_nc_storm_for_climada(ds_track: xr.Dataset) -> xr.Dataset:
    """
    Convert a single-track NC slice into the exact CLIMADA-compatible structure.
    
    Keeps arrays square (padded time steps included), but computes
    start_date and end_date using only valid (non-NaN) core variable time steps.
    """
    # --- Build time coordinate (vectorized) ---
    start_year = int(ds_track["tc_years"].values)
    start_month = int(ds_track["tc_month"].values)
    start_dt = pd.Timestamp(year=start_year, month=start_month, day=1)

    time_seconds = ds_track["time"].values
    time_dt = pd.to_datetime(time_seconds, unit="s", origin=start_dt).to_numpy()
    n_time = time_dt.size

    # --- Identify fully-valid time steps (all core vars finite) and use those
    # to derive start/end date. The track may have leading/trailing steps where
    # only SOME core vars are finite (those survived the upstream any-finite
    # trim in read_single_storm_from_dataset). ---
    core_vars = ["lon", "lat", "max_sustained_wind", "central_pressure", "environmental_pressure"]
    valid_mask = np.all(
        np.stack([~np.isnan(ds_track[v].values) for v in core_vars], axis=0),
        axis=0,
    )
    time_dt_valid = time_dt[valid_mask]
    start_date_iso = str(time_dt_valid[0].astype("datetime64[D]"))
    end_date_iso = str(time_dt_valid[-1].astype("datetime64[D]"))

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
    
def chmod_recursive(path: Path, mode: int = 0o775):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)

def ensure_min_grid(da: xr.DataArray, buffer_deg: float = 0.1) -> xr.DataArray:
    """
    Pad a single-pixel lat or lon dimension with NaN neighbors so downstream
    rasterization always sees at least a 3-cell axis.
    """
    if da.lat.size == 1:
        c = float(da.lat.values[0])
        da = da.reindex(lat=[c-buffer_deg, c, c+buffer_deg], fill_value=np.nan)

    if da.lon.size == 1:
        c = float(da.lon.values[0])
        da = da.reindex(lon=[c-buffer_deg, c, c+buffer_deg], fill_value=np.nan)

    return da


def normalize_lon_to_180(da: xr.DataArray) -> xr.DataArray:
    """Shift longitude coords from 0-360 to -180..180 and sort ascending."""
    return da.assign_coords(
        lon=(((da.lon + 180) % 360) - 180)
    ).sortby("lon")

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
    except ValueError as e:
        raise ValueError(
            f"Windfield shape mismatch for storm {storm_name}: "
            f"got {wf.shape}, expected ({n_time}, {n_lat}, {n_lon}, 2)"
        ) from e

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
        return empty_da

    # ---- For valid storms ----
    # Remove empty space (zeros) outside storm footprint
    yearly_exposure = yearly_exposure.where(yearly_exposure > 0)
    yearly_exposure = yearly_exposure.dropna(dim="lat", how="all")
    yearly_exposure = yearly_exposure.dropna(dim="lon", how="all")


    # --------------------------------------------------
    # Expand single-pixel footprint by creating new grid
    # --------------------------------------------------
    yearly_exposure = ensure_min_grid(yearly_exposure)

    return yearly_exposure


######################################
#    Per Storm Intensity Functions   #
######################################


class NoIntensityError(Exception):
    """Raised when CLIMADA produces zero intensity at every centroid for a storm."""


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
    # Read the scalar storm-level attr (set in normalize_nc_storm_for_climada),
    # NOT the per-timestep `basin` data variable, which would give an array.
    storm_basin = getattr(event, "storm_basin", None)
    storm_category = event.category
    storm_id = event.sid
    start_date = event.start_date
    end_date = event.end_date


    # shape: (n_centroids,)
    intensity_flat = haz.intensity.toarray()[0, :]

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

    if float(da.max().values) == 0:
        raise NoIntensityError(
            f"Storm {storm_name} produced zero intensity at every centroid"
        )

    # ---- Crop to storm footprint in 0–360 ----
    da = da.where(da > 0)
    da = da.dropna(dim="lat", how="all")
    da = da.dropna(dim="lon", how="all")

    # --------------------------------------------------
    # Expand single-pixel footprint by creating new grid
    # --------------------------------------------------
    da = ensure_min_grid(da)

    return da

#############################
#       Helper Functions    #
#############################

def get_storm_indices(ds_all: xr.Dataset) -> list[int]:
    """
    Return a list of storm indices that have at least one timestep where ALL
    core variables are non-NaN. Vectorized across n_trk for speed.
    """
    core_vars = ["lon", "lat", "max_sustained_wind", "central_pressure", "environmental_pressure"]

    # Per (n_trk, time) timestep: True iff every core var is finite.
    # np.stack -> shape (len(core_vars), n_trk, time); reduce over var axis.
    per_timestep_valid = np.all(
        np.stack([~np.isnan(ds_all[v].values) for v in core_vars], axis=0),
        axis=0,
    )
    # Per storm: True iff any timestep is valid.
    has_any_valid = per_timestep_valid.any(axis=1)
    return np.nonzero(has_any_valid)[0].tolist()


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

def load_global_land_gdf(basin: str) -> gpd.GeoDataFrame:
    """
    Load the full basin-appropriate land-polygon GeoDataFrame once per draw.

    - NA basin uses the normalized (-180..180) admin0 shapefile.
    - All other basins use the 0-360 global WGS84 parquet.

    Builds the spatial index eagerly so the first per-storm bbox query
    doesn't pay for it.
    """
    if basin == "NA":
        shp_path = SHP_ROOT_NORMALIZED / "lbd_standard_admin_0.shp"
        gdf = gpd.read_file(shp_path)
    else:
        gdf = gpd.read_parquet(GDF_PATH)

    _ = gdf.sindex  # force the spatial index to build
    return gdf


def subset_land_for_storm(
    global_land_gdf: gpd.GeoDataFrame,
    storm_da: xr.DataArray,
    basin: str,
    buffer: float = 0.25,
) -> gpd.GeoDataFrame:
    """
    Return land polygons intersecting the storm bbox, each clipped to the bbox.
    Uses the pre-built spatial index on the global gdf for an O(log n) candidate
    query, then a precise `intersects` predicate.

    - NA basin: storm coords are -180..180, no wraparound.
    - Other basins: storm coords are 0-360; bbox may straddle the 360 seam,
      handled by splitting into two boxes and unioning.
    """
    min_lon = float(storm_da["lon"].min()) - buffer
    max_lon = float(storm_da["lon"].max()) + buffer
    min_lat = float(storm_da["lat"].min()) - buffer
    max_lat = float(storm_da["lat"].max()) + buffer

    if basin == "NA":
        bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
    elif min_lon < 0:
        bbox_geom = gpd.GeoSeries(
            [
                box(0, min_lat, max_lon, max_lat),
                box(360 + min_lon, min_lat, 360, max_lat),
            ],
            crs=global_land_gdf.crs,
        ).union_all()
    elif max_lon > 360:
        bbox_geom = gpd.GeoSeries(
            [
                box(min_lon, min_lat, 360, max_lat),
                box(0, min_lat, max_lon - 360, max_lat),
            ],
            crs=global_land_gdf.crs,
        ).union_all()
    else:
        bbox_geom = box(min_lon, min_lat, max_lon, max_lat)

    candidate_idx = global_land_gdf.sindex.query(bbox_geom, predicate="intersects")
    if len(candidate_idx) == 0:
        return global_land_gdf.iloc[0:0].copy()

    subset = global_land_gdf.iloc[candidate_idx].copy()
    subset["geometry"] = subset.geometry.intersection(bbox_geom)
    subset = subset[~subset.geometry.is_empty].copy()
    return subset

def check_storm_landfall(
    storm_da: xr.DataArray,
    land_geom,
    wind_threshold: float = 17.0,
) -> bool:
    """
    Determine if a storm makes landfall.

    Returns True only if storm has winds > wind_threshold
    AND those winds intersect land.

    `land_geom` is a pre-unioned land polygon (shapely Polygon or MultiPolygon),
    computed once per storm and reused across landfall + clip calls.

    Uses RasterArray.clip().mask() to handle small islands robustly.
    """

    if float(storm_da.max().values) < wind_threshold:
        return False

    storm_da = ensure_min_grid(storm_da)

    if storm_da.lat.size < 2 or storm_da.lon.size < 2:
        return False

    raster = to_raster(
        ds=storm_da,
        no_data_value=np.nan,
        lat_col="lat",
        lon_col="lon",
        crs="EPSG:4326",
    )

    xmin, xmax, ymin, ymax = raster.bounds
    raster_bbox = box(xmin, ymin, xmax, ymax)

    if not raster_bbox.intersects(land_geom):
        return False  # storm is fully over ocean

    try:
        raster_clipped = raster.clip(land_geom)
    except WindowError:
        return False  # no overlap in pixel space

    if raster_clipped._ndarray.size == 0:
        return False

    try:
        raster_land = raster_clipped.mask(land_geom, all_touched=True)
    except WindowError:
        return False

    if raster_land._ndarray.size == 0:
        return False

    data = raster_land._ndarray
    strong_wind_mask = np.isfinite(data) & (data > wind_threshold)
    return np.any(strong_wind_mask)

def to_raster(
    ds: xr.DataArray,
    no_data_value: float | int,
    lat_col: str = "lat",
    lon_col: str = "lon",
    crs: str = "EPSG:4326",
) -> rt.RasterArray:
    lat, lon = ds[lat_col].data, ds[lon_col].data

    dlat = (lat[1:] - lat[:-1]).mean()
    dlon = (lon[1:] - lon[:-1]).mean()

    # 🔑 detect latitude direction
    lat_increasing = lat[1] > lat[0]

    if lat_increasing:
        # south → north → flip required
        data = ds.data[::-1]
        y_origin = lat[-1]
    else:
        # already north → south → no flip
        data = ds.data
        y_origin = lat[0]

    transform = Affine(
        a=dlon,
        b=0.0,
        c=lon[0],
        d=0.0,
        e=-abs(dlat),
        f=y_origin,
    )

    return rt.RasterArray(
        data=data,
        transform=transform,
        crs=crs,
        no_data_value=no_data_value,
    )



def clip_raster_to_land(
    da: xr.DataArray,
    land_geom,
) -> xr.DataArray:
    """`land_geom` is a pre-unioned land polygon (shapely)."""

    if da.sizes.get("lat", 0) < 2 or da.sizes.get("lon", 0) < 2:
        da = ensure_min_grid(da, buffer_deg=0.1)

    raster = to_raster(
        ds=da,
        no_data_value=np.nan,
        lat_col="lat",
        lon_col="lon",
        crs="EPSG:4326",
    )

    try:
        raster_land = raster.clip(land_geom).mask(land_geom, all_touched=True)
    except Exception:
        return da * np.nan

    # 4️⃣ Build coordinates from transform
    nrows, ncols = raster_land.shape
    transform = raster_land.transform

    lon = transform.c + transform.a * np.arange(ncols)
    lat = transform.f + transform.e * np.arange(nrows)

    # 5️⃣ Back to DataArray
    da_land = xr.DataArray(
        raster_land._ndarray,
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
        name=da.name,
    )

    # 6️⃣ Preserve attrs
    da_land.attrs.update(getattr(da, "attrs", {}))

    return da_land


#######################################
#    Save Per Storm Functions         #
#######################################

def save_single_storm_metric(
    da: xr.DataArray,
    metric: str,
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
    Save a single storm's DataArray for a given metric (e.g. "intensity",
    "exposure_hours") to a draw-level Zarr store, one group per storm.
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
        / metric
        / f"{metric}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
    )

    draw_store.parent.mkdir(parents=True, exist_ok=True)

    storm_key = f"storm_{storm_index:04d}"

    if draw_store.exists():
        z = zarr.open(draw_store, mode="a")
        if storm_key in z:
            print(f"♻️ Overwriting existing storm {storm_index} in {draw_store}")
            del z[storm_key]

    da = da.copy()
    da.name = metric
    if da.dtype != "float32":
        da = da.astype("float32")

    da = da.chunk({"lat": 64, "lon": 64})
    ds = da.to_dataset()
    ds.attrs.update(sanitize_attrs(da.attrs))

    encoding = {
        metric: {
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
        }
    }

    ds.to_zarr(
        draw_store,
        group=storm_key,
        mode="a",
        encoding=encoding,
        zarr_format=3,
        consolidated=False,
    )


def _landfall_parquet_path(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    storm_index: int,
    save_root: Path = SAVE_ROOT,
) -> Path:
    """Path to a single storm's landfall_locations parquet. Mirrors the
    intensity zarr's draw-level directory naming so the layouts align."""
    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")
    return (
        save_root
        / "metadata"
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / "landfall_locations"
        / f"landfall_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}"
        / f"storm_{storm_index:04d}.parquet"
    )


def compute_storm_landfall_locations(
    storm_intensity: xr.DataArray,
    land_gdf: gpd.GeoDataFrame,
    wind_threshold: float = 17.0,
) -> list[dict]:
    """
    For each polygon in `land_gdf`, compute the max wind speed within that
    polygon and return one record per polygon whose max meets the threshold.
    """
    if land_gdf.empty:
        return []

    start_date = getattr(storm_intensity, "start_date", None)

    raster = to_raster(
        ds=storm_intensity,
        no_data_value=np.nan,
        lat_col="lat",
        lon_col="lon",
        crs="EPSG:4326",
    )

    records: list[dict] = []
    for loc_id, adm0_name, geom in zip(
        land_gdf["loc_id"], land_gdf["ADM0_NAME"], land_gdf.geometry,
    ):
        try:
            clipped = raster.clip(geom).mask(geom, all_touched=True)
        except Exception:
            continue
        data = clipped._ndarray
        finite_mask = np.isfinite(data)
        if not finite_mask.any():
            continue
        max_wind = float(data[finite_mask].max())
        if max_wind < wind_threshold:
            continue
        records.append({
            "start_date": start_date,
            "loc_id": loc_id,
            "ADM0_NAME": adm0_name,
            "max_wind_m_s": max_wind,
        })
    return records


def save_storm_landfall_locations(
    records: list[dict],
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    storm_index: int,
    start_date: str | None,
    save_root: Path = SAVE_ROOT,
) -> None:
    """
    Write a single storm's per-location landfall records to parquet. Always
    writes a file (even if `records` is empty) so resume logic can use the
    parquet as the completion sentinel:
      - file absent      → storm not yet processed
      - file empty       → processed, no landfall (no zarrs expected)
      - file has rows    → processed with landfall (zarrs must exist too)
    """
    out_path = _landfall_parquet_path(
        source_id, variant_label, experiment_id, batch_year,
        basin, draw, storm_index, save_root,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "source_id": source_id,
            "variant_label": variant_label,
            "experiment_id": experiment_id,
            "batch_year": batch_year,
            "basin": basin,
            "draw": draw,
            "storm_id": storm_index,
            "start_date": start_date,
            **r,
        }
        for r in records
    ]
    df = pd.DataFrame(rows, columns=[
        "source_id", "variant_label", "experiment_id", "batch_year",
        "basin", "draw", "storm_id", "start_date", "loc_id", "ADM0_NAME", "max_wind_m_s",
    ])
    df.to_parquet(out_path, index=False)



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

    metrics = ["intensity", "exposure_hours"]

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
    """
    Decide whether a storm is fully processed and can be skipped on resume.

    Sentinel: the landfall parquet.
      - missing  → storm not yet processed
      - empty    → processed, no landfall (zarrs are not expected)
      - has rows → processed with landfall (zarrs must also exist + validate)
    """
    landfall_path = _landfall_parquet_path(
        source_id, variant_label, experiment_id, batch_year,
        basin, draw, storm_index, save_root,
    )
    if not landfall_path.exists():
        return False

    try:
        landfall_df = pd.read_parquet(landfall_path)
    except Exception as e:
        print(f"⚠️ Could not read landfall parquet {landfall_path}: {e}; will reprocess.")
        landfall_path.unlink(missing_ok=True)
        return False

    if landfall_df.empty:
        return True  # no-landfall sentinel; no zarrs to validate

    metrics = ["intensity", "exposure_hours"]
    expected_arrays = {
        "intensity": {"intensity", "lat", "lon"},
        "exposure_hours": {"exposure_hours", "lat", "lon"},
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

        if any(storm_path.rglob("*.partial")):
            shutil.rmtree(storm_path, ignore_errors=True)
            print(f"⚠️ Deleted corrupted {storm_key} in {metric}")
            return False

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
        "completed_utc": datetime.now(timezone.utc).isoformat(),
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
        
def process_single_storm(
    storm_index, ds_all, save_root,
    source_id, variant_label, experiment_id, batch_year, basin, draw,
    centroids, global_land_gdf,
) -> None:
    """
    Process a single storm: validate existing zarr, compute intensity + exposure,
    clip to land, and persist both metrics. Errors propagate to the caller, which
    catches and continues to the next storm.
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
    # start_date is fixed once tc_tracks is built and survives even if
    # downstream CLIMADA processing raises.
    start_date = tc_tracks.data[0].attrs.get("start_date")

    # Helper to write the empty-parquet sentinel for no-landfall storms.
    def _mark_no_landfall(reason: str) -> None:
        print(f"⚠️ Storm {storm_index} {reason}; writing empty landfall marker.")
        save_storm_landfall_locations(
            [],
            source_id=source_id,
            variant_label=variant_label,
            experiment_id=experiment_id,
            batch_year=batch_year,
            basin=basin,
            draw=draw,
            storm_index=storm_index,
            start_date=start_date,
            save_root=save_root,
        )

    haz = generate_hazard_per_track(tc_tracks, centroids)

    try:
        storm_intensity = generate_intensity_per_storm(haz, centroids, tc_tracks)
    except NoIntensityError as e:
        # CLIMADA produced no intensity output for this storm — treat as no
        # landfall so future resumes short-circuit before re-running CLIMADA.
        _mark_no_landfall(f"has no CLIMADA intensity ({e})")
        return

    if basin == "NA":
        storm_intensity = normalize_lon_to_180(storm_intensity)

    land_gdf = subset_land_for_storm(
        global_land_gdf=global_land_gdf,
        storm_da=storm_intensity,
        basin=basin,
        buffer=0.25,
    )

    if land_gdf.empty:
        _mark_no_landfall("has no land polygons in bbox")
        return

    # Compute the unioned land geometry once and reuse for landfall check
    # and both clip_raster_to_land calls (intensity + exposure).
    land_geom = land_gdf.geometry.union_all()

    if not check_storm_landfall(storm_intensity, land_geom, 17):
        _mark_no_landfall("does not make landfall")
        return

    storm_intensity = clip_raster_to_land(storm_intensity, land_geom)

    save_single_storm_metric(
        storm_intensity,
        metric="intensity",
        storm_index=storm_index,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        save_root=save_root,
    )

    # Per-location landfall records (uses the same land-clipped intensity)
    landfall_records = compute_storm_landfall_locations(
        storm_intensity, land_gdf, wind_threshold=17.0,
    )
    save_storm_landfall_locations(
        landfall_records,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        storm_index=storm_index,
        start_date=start_date,
        save_root=save_root,
    )

    del storm_intensity

    storm_speed = generate_speed_per_storm(haz, centroids, tc_tracks)
    storm_exposure = compute_yearly_exposure_per_storm(
        storm_speed,
        wind_threshold=17.0,
    )

    if basin == "NA":
        storm_exposure = normalize_lon_to_180(storm_exposure)

    storm_exposure = clip_raster_to_land(storm_exposure, land_geom)

    save_single_storm_metric(
        storm_exposure,
        metric="exposure_hours",
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

    # Load the basin-appropriate land polygons once per draw; the per-storm
    # subset uses the gdf's spatial index for fast bbox queries.
    global_land_gdf = load_global_land_gdf(basin)

    # Open once for entire draw. check_existing_storm_in_zarr inside
    # process_single_storm validates each storm's structure and skips
    # (or cleans up) already-completed storms.
    with xr.open_dataset(nc_file) as ds_all:

        storm_indices = get_storm_indices(ds_all)

        centroids = generate_basin_centroids(basin, res=RESOLUTION)

        for storm_index in storm_indices:
            try:
                process_single_storm(
                    storm_index,
                    ds_all,
                    save_root,
                    source_id,
                    variant_label,
                    experiment_id,
                    batch_year,
                    basin,
                    draw,
                    centroids,
                    global_land_gdf,
                )
            except Exception as e:
                print(
                    f"❌ Storm {storm_index} failed in draw {draw} "
                    f"({source_id}/{variant_label}/{experiment_id}/{batch_year}/{basin}): "
                    f"{type(e).__name__}: {e}"
                )
                traceback.print_exc()
                gc.collect()
                continue


    # finalize permissions once per draw on the zarr stores we actually wrote
    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")
    basin_dir = (
        save_root / source_id / variant_label / experiment_id / batch_year / basin
    )

    for metric in ("intensity", "exposure_hours"):
        zarr_path = (
            basin_dir / metric
            / f"{metric}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
        )
        if zarr_path.exists():
            try:
                chmod_recursive(zarr_path, mode=0o775)
            except Exception as e:
                print(f"⚠️ Could not set permissions for {zarr_path}: {e}")

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