from pathlib import Path
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
from climada.hazard import TCTracks, TropCyclone, Centroids
import gc
import re
import rasterra as rt # type: ignore
import datetime as dt
import pandas as pd  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
import geopandas as gpd  # type: ignore
from shapely.geometry import Point, box, mapping  # type: ignore
from affine import Affine  # type: ignore
import os
import argparse

parser = argparse.ArgumentParser(description="Run James code")

# Define arguments
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--draw", type=int, required=True, help="Draw index")
parser.add_argument("--storm_index", type=int, default=None, help="Storm Index")
parser.add_argument("--sample_name", type=str, required=True, help="Sample Name")
parser.add_argument("--relative_risk", type=str, required=True, help="Relative Risk Type (rd or cvd)")


# Parse arguments
args = parser.parse_args()
source_id = args.source_id
variant_label = args.variant_label
experiment_id = args.experiment_id
batch_year = args.batch_year
basin = args.basin
draw = args.draw
storm_index = args.storm_index
sample_name = args.sample_name
relative_risk = args.relative_risk

# Constants
ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")
RESOLUTION = 0.1  # degrees


def read_custom_tracks(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
) -> Path:
    """
    Locate TC risk Zarr draw for CLIMADA processing.

    Returns
    -------
    Path
        Path to the draw-level Zarr store.
    """

    # temp
    start_year, end_year = batch_year.split("-")

    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    track_store = (
        ROOT_PATH
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"tracks_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
    )

    if not track_store.exists():
        raise FileNotFoundError(f"Zarr store not found: {track_store}")

    return track_store



def read_storm_groups_from_draw(draw_store: Path) -> list[xr.Dataset]:
    """
    Discover and open all storm_* Zarr groups in a draw.

    Parameters
    ----------
    draw_store : Path
        Path to draw-level Zarr store.

    Returns
    -------
    List[xr.Dataset]
        List of storm-level xarray Datasets (raw, unmodified).
    """
    if not draw_store.exists():
        raise FileNotFoundError(f"Draw store not found: {draw_store}")

    storm_paths = sorted(
        p for p in draw_store.iterdir()
        if p.is_dir() and p.name.startswith("storm_")
    )

    storms = []
    for storm_path in storm_paths:
        ds_storm = xr.open_zarr(storm_path, consolidated=False)
        storms.append(ds_storm)

    return storms


def read_single_storm_from_draw(draw_store: Path, storm_index: int) -> xr.Dataset:
    """
    Open a single storm_* Zarr group from a draw.

    Parameters
    ----------
    draw_store : Path
        Path to the draw-level Zarr store.
    storm_index : int
        Storm index (0..N), corresponds to 'storm_0000', 'storm_0001', etc.

    Returns
    -------
    xr.Dataset
        Single storm dataset (raw, unmodified).
    """
    if not draw_store.exists():
        raise FileNotFoundError(f"Draw store not found: {draw_store}")

    # Build storm folder name
    storm_name = f"storm_{storm_index:04d}"
    storm_path = draw_store / storm_name

    if not storm_path.exists():
        raise FileNotFoundError(f"Storm Zarr not found: {storm_path}")

    # Open single storm dataset lazily
    ds_storm = xr.open_zarr(storm_path, consolidated=False)

    return ds_storm


def normalize_storm_for_climada(ds_storm: xr.Dataset) -> xr.Dataset:
    """
    Normalize a raw Zarr storm dataset into the exact CLIMADA-compatible
    storm Dataset expected by the existing pipeline.
    """

    # --- time coordinate ---
    time_dt = ds_storm.coords["time"].values
    n_time = time_dt.size

    # --- core variables ---
    lon = ds_storm["lon"].values
    lat = ds_storm["lat"].values
    vmax = ds_storm["max_sustained_wind"].values
    cp = ds_storm["central_pressure"].values
    env = ds_storm["environmental_pressure"].values
    basin = ds_storm["basin"].values

    # Safety: normalize longitude if needed
    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180

    # --- time step (already provided, hourly) ---
    dt_hours = float(ds_storm["time_step"].values[0])

    # --- attributes ---
    start_date_iso = (
        time_dt[0].astype("datetime64[D]").item().isoformat()
    )
    end_date_iso = (
        time_dt[-1].astype("datetime64[D]").item().isoformat()
    )

    storm_name = ds_storm.attrs.get("name", "unnamed_storm")
    storm_basin = basin[0]

    sid = int(ds_storm.attrs.get("sid", -1))
    id_no = int(ds_storm.attrs.get("id_no", sid))
    category = int(ds_storm.attrs.get("category", 0))

    # --- construct final dataset ---
    ds = xr.Dataset(
        coords={"time": time_dt},
        data_vars={
            "lon": (("time",), lon),
            "lat": (("time",), lat),
            "max_sustained_wind": (("time",), vmax),
            "central_pressure": (("time",), cp),
            "environmental_pressure": (("time",), env),
            "basin": (("time",), basin),
            # kept for backward compatibility
            "radius_max_wind": (("time",), np.zeros(n_time)),
            "radius_oci": (("time",), np.zeros(n_time)),
            "time_step": (("time",), np.full(n_time, dt_hours)),
        },
        attrs={
            "name": storm_name,
            "start_date": start_date_iso,
            "end_date": end_date_iso,
            "storm_basin": storm_basin,
            "sid": sid,
            "id_no": id_no,
            "category": category,
            "orig_event_flag": True,
            "data_provider": "custom",
            "max_sustained_wind_unit": "kn",
            "central_pressure_unit": "mb",
        },
    )

    return ds


def prepare_zarr_files(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    storm_index: int | str = None,
) -> "TCTracks":
    """
    Prepare CLIMADA TCTracks from a Zarr draw.

    Parameters
    ----------
    root_path : Path
        Base path to CMIP6 input.
    source_id : str
        Model source id, e.g., 'ACCESS-CM2'.
    variant_label : str
        Variant label, e.g., 'r1i1p1f1'.
    experiment_id : str
        Experiment id, e.g., 'historical'.
    batch_year : str
        Batch year string, e.g., '1970-1989'.
    basin : str
        Basin code, e.g., 'EP'.
    draw : int
        Draw index, e.g., 0..98.

    Returns
    -------
    TCTracks
        CLIMADA-compatible TCTracks object containing all storms from the draw.
    """

    # --- 1. Locate the draw-level Zarr store ---
    draw_store = read_custom_tracks(
        source_id,
        variant_label,
        experiment_id,
        batch_year,
        basin,
        draw,
    )

    # --- 2. Read all storms ---
    if storm_index is not None:
        raw_storm = read_single_storm_from_draw(draw_store, storm_index)
        storms = [normalize_storm_for_climada(raw_storm)]
        tc_tracks = TCTracks(data=storms)
    else:
        raw_storms: list[xr.Dataset] = read_storm_groups_from_draw(draw_store)
        storms = [normalize_storm_for_climada(ds) for ds in raw_storms]
        tc_tracks = TCTracks(data=storms)

    return tc_tracks



############################################
#              Helper Functions            #
############################################
    
def normalize_lon(lon: float) -> float:
    """Normalize longitude to [-180, 180] range."""
    lon = ((lon + 180) % 360) - 180
    return lon


######################################
#    Basin Centroid Functions        #
######################################

def generate_basin_centroids(basin: str, res: float = 0.1) -> "Centroids":
    """
    Generate Centroids for a specific tropical cyclone basin.
    """

    # Dictionary of basin bounds
    basin_bounds = {
        'EP': ['180E', '0N', '290E', '60N'],
        'NA': ['260E', '0N', '360E', '60N'],
        'NI': ['30E', '0N', '100E', '50N'],
        'SI': ['20E', '45S', '100E', '0S'],
        'AU': ['100E', '45S', '180E', '0S'],
        'SP': ['180E', '45S', '250E', '0S'], # Original SA - possible mismatch
        'WP': ['100E', '0N', '180E', '60N'],
        'GL': ['0E', '90S', '360E', '90N']
    }

    if basin not in basin_bounds:
        raise ValueError(f"Basin '{basin}' not recognized. Available: {list(basin_bounds.keys())}")

    def parse_coord(coord_str: str) -> float:
        """Convert coordinate string with direction to float degrees."""
        match = re.match(r"([0-9\.]+)([ENWS])", coord_str)
        if not match:
            raise ValueError(f"Invalid coordinate string: {coord_str}")
        val, dir_ = match.groups()
        val = float(val)
        if dir_ in ['W', 'S']:
            val = -val
        return val

    lon_min, lat_min, lon_max, lat_max = [parse_coord(c) for c in basin_bounds[basin]]

    # Normalize longitudes to [-180, 180]
    lon_min = normalize_lon(lon_min)
    lon_max = normalize_lon(lon_max)

    # Expand upper bounds by resolution to include last grid cell
    lon_max += res
    lat_max += res

    # Create Centroids for the basin
    centroids = Centroids.from_pnt_bounds((lon_min, lat_min, lon_max, lat_max), res=res)

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

def generate_speed_per_storm(haz: TropCyclone, centroids: Centroids, tc_tracks: TCTracks) -> list[xr.DataArray]:
    """
    Generate per-storm wind speed DataArrays for a list of tropical cyclones.
    """

    lat = np.unique(centroids.coord[:, 0])
    lon = np.unique(centroids.coord[:, 1])
    lat = np.sort(lat)

    storm_list_speed = []

    for i, event in enumerate(tc_tracks.data):
        storm_name = event.name  # <-- Grab the storm name directly
        storm_start_date = event.start_date
        storm_end_date = event.end_date
        storm_basin = event.storm_basin
        storm_category = event.category
        times = event.time  # array of timesteps
        wf = haz.windfields[i].toarray()  # shape: (time, n_centroids, 2)

        n_time = len(times)
        n_lat = len(lat)
        n_lon = len(lon)

        try:
            wf_reshaped = wf.reshape(n_time, n_lat, n_lon, 2)
        except ValueError:
            print(f"⚠️ Skipping storm {storm_name} due to shape mismatch")
            continue

        # Preserve timestep if it exists
        timestep = getattr(event, "time_step", None)
        coords = {"time": times, "lat": np.flip(lat), "lon": lon, "dir": ["u", "v"]}
        if timestep is not None:
            coords["time_step"] = ("time", np.array(timestep))


        da = xr.DataArray(
            wf_reshaped,
            coords=coords,
            dims=["time", "lat", "lon", "dir"],
            name=f"{storm_name}_windfields"
        )

        # Compute wind speed
        da_speed = np.sqrt(da.isel(dir=0)**2 + da.isel(dir=1)**2)
        da_speed.name = storm_name  # <-- Name the DataArray after the storm
        da_speed.attrs.update({
            "description": f"Storm {storm_name} wind speed",
            "units": "m/s",
            "storm_name": storm_name,
            "start_date": storm_start_date,
            "end_date": storm_end_date,
            "basin": storm_basin,
            "category": storm_category,
        })

        storm_list_speed.append(da_speed)

        # Free memory
        del wf, wf_reshaped, da, da_speed

    return storm_list_speed

######################################
#    Per Storm Intensity Functions   #
######################################

def generate_intensity_per_storm(
    haz: TropCyclone,
    centroids: Centroids,
    tc_tracks: TCTracks,
) -> list[xr.DataArray]:
    """
    Generate per-storm, per-pixel intensity using CLIMADA haz.intensity.

    Intensity is defined as the maximum wind speed experienced at each pixel
    during the storm lifetime.

    Returns
    -------
    list[xr.DataArray]
        One DataArray per storm with dims ('lat', 'lon')
    """

    lat = np.unique(centroids.coord[:, 0])
    lon = np.unique(centroids.coord[:, 1])
    lat = np.sort(lat)

    n_lat = len(lat)
    n_lon = len(lon)

    storm_list_intensity = []

    for i, event in enumerate(tc_tracks.data):

        storm_name = event.name
        storm_basin = event.storm_basin
        storm_category = event.category
        storm_start = event.start_date
        storm_end = event.end_date

        try:
            # shape: (n_centroids,)
            intensity_flat = haz.intensity.toarray()[i, :]
        except Exception as e:
            print(f"⚠️ Could not read intensity for {storm_name}: {e}")
            continue

        if intensity_flat.size != n_lat * n_lon:
            print(
                f"⚠️ Skipping {storm_name}: grid mismatch "
                f"{intensity_flat.size} vs {n_lat * n_lon}"
            )
            continue

        # reshape to 2D grid
        intensity_2d = np.flip(
            intensity_flat.reshape(n_lat, n_lon),
            axis=0
        )

        da = xr.DataArray(
            intensity_2d,
            coords={"lat": lat[::-1], "lon": lon},
            dims=["lat", "lon"],
            name=f"{storm_name}_intensity",
        )

        da.attrs.update({
            "description": "Per-storm pixel-level maximum wind speed",
            "units": "m/s",
            "storm_name": storm_name,
            "start_date": storm_start,
            "end_date": storm_end,
            "basin": storm_basin,
            "category": storm_category,
            "definition": (
                "Maximum wind speed experienced at each pixel "
                "during the storm lifetime"
            ),
        })

        storm_list_intensity.append(da)

    return storm_list_intensity


#############################
#    Reading in Functions   #
#############################

def load_relative_risk_df(relative_risk: str,root: Path = Path("/mnt/share/homes/mfiking/github_repos/climada_python/data/")):
    relative_risk_df = pd.read_csv(root / f"{relative_risk}_rr_samples.csv")

    return relative_risk_df

def load_shapefiles():
    shapefile=gpd.read_parquet('/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/shapes_gbd_2025.parquet')

    return shapefile

def load_in_gridded_population(year: int | str, meters: int | str, bounds: tuple):

    pop_raster = rt.load_raster(f"/mnt/team/rapidresponse/pub/population-model/results/current/world_cylindrical_{meters}/{year}q1.tif", 
                                bounds=(bounds[0], bounds[1], bounds[2], bounds[3]))
    
    return pop_raster


#############################
#       Helper Functions    #
#############################

def generate_basin_bounds_54034(basin: str) -> tuple:
    """
    Generate ESRI:54034 basin bounds using the same basin definitions
    and coordinate parsing logic as generate_basin_centroids.

    Parameters
    ----------
    basin : str
        Basin code (e.g., 'EP', 'NA').

    Returns
    -------
    tuple
        (xmin, ymin, xmax, ymax) in ESRI:54034
    """

    # Dictionary of basin bounds (identical to centroid function)
    basin_bounds = {
        'EP': ['180E', '0N', '290E', '60N'],
        'NA': ['260E', '0N', '360E', '60N'],
        'NI': ['30E', '0N', '100E', '50N'],
        'SI': ['20E', '45S', '100E', '0S'],
        'AU': ['100E', '45S', '180E', '0S'],
        'SP': ['180E', '45S', '250E', '0S'],
        'WP': ['100E', '0N', '180E', '60N'],
        'GL': ['0E', '90S', '360E', '90N']
    }

    if basin not in basin_bounds:
        raise ValueError(
            f"Basin '{basin}' not recognized. Available: {list(basin_bounds.keys())}"
        )

    def parse_coord(coord_str: str) -> float:
        """Convert coordinate string with direction to float degrees."""
        match = re.match(r"([0-9\.]+)([ENWS])", coord_str)
        if not match:
            raise ValueError(f"Invalid coordinate string: {coord_str}")
        val, dir_ = match.groups()
        val = float(val)
        if dir_ in ['W', 'S']:
            val = -val
        return val

    # Parse bounds
    lon_min, lat_min, lon_max, lat_max = [
        parse_coord(c) for c in basin_bounds[basin]
    ]

    # Normalize longitudes exactly as in centroid logic
    lon_min = normalize_lon(lon_min)
    lon_max = normalize_lon(lon_max)

    # Build basin polygon in EPSG:4326
    basin_gdf = gpd.GeoDataFrame(
        geometry=[box(lon_min, lat_min, lon_max, lat_max)],
        crs="EPSG:4326",
    )

    # Reproject to ESRI:54034
    basin_gdf_54034 = basin_gdf.to_crs("ESRI:54034")

    # Return numeric bounds
    xmin, ymin, xmax, ymax = basin_gdf_54034.total_bounds
    return xmin, ymin, xmax, ymax


def knots_to_ms(knots):
    """
    Convert wind speed from knots to meters per second.
    
    Parameters:
    -----------
    knots : float, array-like, or xarray.DataArray
        Wind speed in knots
        
    Returns:
    --------
    float, array-like, or xarray.DataArray
        Wind speed in meters per second
        
    Notes:
    ------
    Conversion factor: 1 knot = 0.514444 m/s
    """
    return knots * 0.514444

def interpolate_rr_from_windspeed(intensity_array, rr_samples_df, sample_name, min_windspeed_knots=25):
    """
    Interpolate relative risk values for windspeed intensity array using a specific sample.
    
    Parameters:
    -----------
    intensity_array : xarray.DataArray
        Wind intensity values in m/s
    rr_samples_df : pandas.DataFrame
        DataFrame with windspeed (knots), type, and sample columns
    sample_name : str
        Name of the sample column to use (e.g., 'sample_001')
    min_windspeed_knots : float
        Minimum windspeed threshold in knots (default: 25)
        
    Returns:
    --------
    xarray.DataArray
        Relative risk values interpolated for the intensity array
    """
    
    # Convert minimum windspeed to m/s for comparison
    min_windspeed_ms = knots_to_ms(min_windspeed_knots)
    
    # Get windspeed and RR values from the sample
    windspeed_knots = rr_samples_df['windspeed'].values
    windspeed_ms = knots_to_ms(windspeed_knots)
    rr_values = rr_samples_df[sample_name].values
    
    # Create interpolation function
    rr_interp = interp1d(
        windspeed_ms, 
        rr_values, 
        kind='linear', 
        bounds_error=False, 
        fill_value='extrapolate'
    )
    
    # Create copy to preserve coordinates and metadata
    result = intensity_array.copy()
    
    # Get min and max windspeed values from RR data
    min_rr_windspeed_ms = windspeed_ms.min()
    max_rr_windspeed_ms = windspeed_ms.max()
    max_rr_value = rr_values[np.argmax(windspeed_ms)]  # RR value at highest windspeed
    
    # Initialize all values to 0
    rr_interpolated = np.zeros_like(intensity_array.values)
    
    # Create masks for different windspeed ranges
    below_min_mask = intensity_array.values < min_rr_windspeed_ms
    above_max_mask = intensity_array.values > max_rr_windspeed_ms
    interpolation_mask = (intensity_array.values >= min_rr_windspeed_ms) & (intensity_array.values <= max_rr_windspeed_ms)
    
    # Set values below minimum to 0 (already initialized to 0)
    # rr_interpolated[below_min_mask] = 0  # Already 0
    
    # Set values above maximum to the highest RR value
    if np.any(above_max_mask):
        rr_interpolated[above_max_mask] = max_rr_value
    
    # Interpolate values within the RR data range
    if np.any(interpolation_mask):
        rr_values_interp = rr_interp(intensity_array.values[interpolation_mask])
        rr_interpolated[interpolation_mask] = rr_values_interp
    
    # Update the data array values
    result.values = rr_interpolated
    result.name = f"relative_risk_{sample_name}"
    
    return result
    
def intersect_shapefile_with_rr_data(shapefile_gdf, rr_sample, buffer=0):
    """
    Find shapefile rows that intersect with the relative risk data grid.

    Parameters
    ----------
    shapefile_gdf : geopandas.GeoDataFrame
        Shapefile already in target CRS
    rr_sample : xarray.DataArray
        2D relative risk raster (projected or geographic)
    buffer : float, optional
        Buffer distance in CRS units (meters if projected, degrees if geographic)

    Returns
    -------
    geopandas.GeoDataFrame
        Subset of shapefile that intersects RR data
    """

    # Identify spatial dims robustly
    x_dim, y_dim = rr_sample.rio.x_dim, rr_sample.rio.y_dim

    # Mask valid RR cells
    valid_mask = (rr_sample > 0) & rr_sample.notnull()

    # Extract coordinates
    xs, ys = np.meshgrid(
        rr_sample[x_dim].values,
        rr_sample[y_dim].values,
        indexing="xy"
    )

    valid_xs = xs[valid_mask.values]
    valid_ys = ys[valid_mask.values]

    print(f"Found {len(valid_xs)} non-zero RR grid cells")

    # Build GeoDataFrame of points in RR CRS
    data_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(valid_xs, valid_ys),
        crs=rr_sample.rio.crs,
    )

    # Optional buffer (units follow CRS!)
    if buffer > 0:
        data_gdf.geometry = data_gdf.geometry.buffer(buffer)
        print(f"Applied buffer of {buffer} CRS units")

    # CRS consistency check
    if shapefile_gdf.crs != data_gdf.crs:
        raise ValueError(
            f"CRS mismatch: shapefile={shapefile_gdf.crs}, rr={data_gdf.crs}"
        )

    # Spatial join
    intersections = gpd.sjoin(
        shapefile_gdf,
        data_gdf,
        how="inner",
        predicate="intersects",
    )

    result = shapefile_gdf.loc[intersections.index.unique()].copy()

    print(f"Found {len(result)} intersecting admin units")

    return result



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

def subset_affected_area(
    rr_raster: rt.RasterArray,
    threshold: float = 0.0,
) -> rt.RasterArray:
    """
    Subset a RasterArray to the minimal bounding box
    where RR > threshold, using rasterra.clip().

    Parameters
    ----------
    rr_raster : RasterArray
        Storm relative risk raster.
    threshold : float
        Threshold defining affected pixels.

    Returns
    -------
    RasterArray
        Subset raster clipped to affected area.
    """
    data = np.asarray(rr_raster.data)

    mask = np.isfinite(data) & (data > threshold)
    if not np.any(mask):
        raise ValueError("No affected pixels found (RR > threshold).")

    rows, cols = np.where(mask)

    transform = rr_raster.transform
    a, b, c, d, e, f = transform[:6]

    # Pixel → coordinate conversion
    xmin = c + cols.min() * a
    xmax = c + (cols.max() + 1) * a
    ymax = f + rows.min() * e
    ymin = f + (rows.max() + 1) * e

    # Build geometry
    geom = box(xmin, ymin, xmax, ymax)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs=rr_raster.crs)
    gdf["geometry"] = gdf.buffer(50_000)  # 50 km buffer


    # Native rasterra clip
    return rr_raster.clip(gdf)



############################################
# Relative Risk Calculation per Storm     #
############################################


def generate_relative_risk_per_storm(
    storm_list_intensity: list[xr.DataArray],
    rr_samples_df,
    sample_name: str,
    min_windspeed_knots: float = 25.0,
) -> list[xr.DataArray]:
    """
    Generate per-storm pixel-level relative risk from storm intensity.

    Parameters
    ----------
    storm_list_intensity : list[xr.DataArray]
        Each DataArray has dims ('lat', 'lon') and values in m/s.
        Represents per-pixel maximum wind speed during the storm.
    rr_samples_df : pandas.DataFrame
        Relative risk lookup table with 'windspeed' column in knots
        and sample columns (e.g. 'sample_001').
    sample_name : str
        Column name in rr_samples_df to use.
    min_windspeed_knots : float
        Minimum windspeed threshold below which RR = 0.

    Returns
    -------
    list[xr.DataArray]
        One DataArray per storm with dims ('lat', 'lon').
    """

    storm_list_rr = []

    for da_intensity in storm_list_intensity:

        storm_name = da_intensity.attrs.get("storm_name", da_intensity.name)

        # Interpolate RR from windspeed (Katrina logic)
        da_rr = interpolate_rr_from_windspeed(
            intensity_array=da_intensity,
            rr_samples_df=rr_samples_df,
            sample_name=sample_name,
            min_windspeed_knots=min_windspeed_knots,
        )

        da_rr.attrs.update({
            "description": (
                "Pixel-level relative risk derived from storm maximum wind speed"
            ),
            "storm_name": storm_name,
            "start_date": da_intensity.attrs.get("start_date"),
            "end_date": da_intensity.attrs.get("end_date"),
            "basin": da_intensity.attrs.get("basin"),
            "category": da_intensity.attrs.get("category"),
            "rr_sample": sample_name,
            "min_windspeed_knots": min_windspeed_knots,
            "definition": (
                "Relative risk interpolated from windspeed using empirical RR curves; "
                "intensity is maximum per-pixel wind speed during storm lifetime"
            ),
        })

        storm_list_rr.append(da_rr)

    return storm_list_rr



####################################################
#       Intersect Shapefiles with Relative Risk    #
####################################################

def generate_intersected_shapefiles_per_storm(
    shapefile_gdf: gpd.GeoDataFrame,
    storm_list_rr: list[xr.DataArray],
    buffer_degrees: float = 0.0,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Generate per-storm GeoDataFrames of admin units intersecting
    relative risk grids.

    Parameters
    ----------
    shapefile_gdf : geopandas.GeoDataFrame
        Admin unit shapefile
    storm_list_rr : list of xr.DataArray
        Per-storm relative risk DataArrays (2D lat/lon)
        Each DataArray must have 'storm_name' and 'start_date' attributes
    buffer_degrees : float, optional
        Buffer applied to RR grid points before intersection

    Returns
    -------
    dict[str, geopandas.GeoDataFrame]
        Dictionary keyed by "{storm_name}_{year}"
    """

    storm_shapes = {}

    for da_rr in storm_list_rr:

        # ---- Extract storm metadata ----
        storm_name = da_rr.attrs.get("storm_name", da_rr.name)
        start_date = da_rr.attrs.get("start_date", None)

        if start_date is None:
            raise ValueError(
                f"Cannot determine year for storm {storm_name}. "
                "Expected 'start_date' attribute."
            )

        year = pd.to_datetime(start_date).year
        storm_key = f"{storm_name}"

        print(f"🌪️ Processing intersections for {storm_key}")

        # ---- Intersect shapefile with RR grid ----
        intersected_gdf = intersect_shapefile_with_rr_data(
            shapefile_gdf,
            da_rr,
            buffer_degrees=buffer_degrees,
        )

        if intersected_gdf.empty:
            print(f"⚠️ No intersections found for {storm_key}")
            continue

        # Attach storm metadata
        intersected_gdf = intersected_gdf.copy()
        intersected_gdf["storm_name"] = storm_name
        storm_shapes[storm_name] = intersected_gdf

    return storm_shapes

############################################
#           Raster PAF Calculations        #
############################################

def compute_paf_raster_per_admin(
    intersecting_shapes: gpd.GeoDataFrame,
    rr_raster: rt.RasterArray,
    pop_raster: rt.RasterArray,
    location_id_col: str = "location_id",
    days_impact: int = 20,
):
    """
    Compute PAF for each admin unit given fine-resolution population and RR rasters.
    """

    # Reproject admin shapes to raster CRS
    if intersecting_shapes.crs != pop_raster.crs:
        intersecting_shapes = intersecting_shapes.to_crs(pop_raster.crs)

    results = []

    for _, row in intersecting_shapes.iterrows():
        location_id = row[location_id_col]
        geom = row.geometry
        # geom = geom.buffer(7500)  # buffer by x meters (adjust as needed)


        # ---- Clip population raster ----
        pop_clip = pop_raster.clip(geom).mask(geom, fill_value=0)
        pop_vals_total = pop_clip._ndarray
        pop_total = np.nansum(pop_vals_total) # total population in admin unit

        
        if pop_clip.size == 0 or pop_total == 0:
            # results.append({"location_id": location_id, "paf": 0.0, "pop_total": pop_total.item()})
            results.append({"location_id": location_id, "paf": 0.0})
            continue

        # ---- Clip RR raster ----
        rr_clip = rr_raster.clip(geom).mask(geom, fill_value=0)
        rr_vals = rr_clip._ndarray

        # ---- Reclip poplation to RR shape ----
        rr_bounds = rr_clip.bounds
        rr_bbox = gpd.GeoDataFrame(
            geometry=[box(*rr_bounds)],
            crs=rr_clip.crs
        )

        # Clip population raster to RR extent
        pop_rr_clip = pop_clip.clip(rr_bbox)
        pop_vals = pop_rr_clip._ndarray


        # ---- Valid pixels: population present & RR > 1 & finite ----
        if pop_total == 0:
            paf_val = 0.0
        else:
            # 2️⃣ Affected pixels:
            #    population present AND RR > 1 AND finite
            affected_mask = (
                (pop_vals > 0) &
                np.isfinite(rr_vals) &
                (rr_vals > 1)
            )

            # 3️⃣ If no affected people → PAF = 0
            if not np.any(affected_mask):
                paf_val = 0.0
            else:
                # 4️⃣ Compute PAF only on affected pixels
                paf_raw = np.zeros_like(rr_vals, dtype=float)
                paf_raw[affected_mask] = (
                    (rr_vals[affected_mask] - 1) / rr_vals[affected_mask]
                    * days_impact / 365
                )

                # 5️⃣ Population-weighted PAF
                paf_weighted = paf_raw * pop_vals

                # Replace NaN / inf with 0 BEFORE summation
                paf_weighted = np.where(np.isfinite(paf_weighted), paf_weighted, 0)

                paf_val = paf_weighted.sum() / pop_total

        # results.append({"location_id": location_id, "paf": paf_val, "pop_total": pop_total.item()})
        results.append({"location_id": location_id, "paf": paf_val})

    return pd.DataFrame(results)


############################################
#              PAF                         #
############################################

def generate_pafs(storm_list_intensity, sample_name, basin, relative_risk):

    # read in shapefiles
    shapefile = load_shapefiles()

    # convert to ESRI:54034
    shapefile = shapefile.to_crs("ESRI:54034")

    # read in relative risk data
    relative_risk_df = load_relative_risk_df(relative_risk)

    # generate storm list of relative risks
    storm_list_rr = generate_relative_risk_per_storm(
        storm_list_intensity,
        relative_risk_df,
        sample_name,
        min_windspeed_knots=25
    )

    # generate basin bounds in ESRI:54034
    xmin, ymin, xmax, ymax = generate_basin_bounds_54034(basin)

    # read in full population raster for the basin and years as a dictionary
    years = []
    for storm in storm_list_intensity:
        year = pd.to_datetime(storm.attrs["start_date"]).year
        if year not in years:
            years.append(year)

    pop_dict = {}
    for year in years:
        pop_raster = load_in_gridded_population(year=year, meters=100, bounds=(xmin, ymin, xmax, ymax))
        pop_dict[year] = pop_raster

    # create empty dataframe to store location_id and paf results
    paf_df_list = []

    for storm_rr in storm_list_rr:

        # get year from storm start date
        year = pd.to_datetime(storm_rr.start_date).year

        # attach CRS if missing
        da_rr = storm_rr
        if not da_rr.rio.crs:
            da_rr = da_rr.rio.write_crs("EPSG:4326")
            da_rr = da_rr.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

        # reproject to ESRI:54034
        da_rr_54034 = da_rr.rio.reproject("ESRI:54034")

        # get intersecting admin unit shapes over a single storm
        intersecting_shapes = intersect_shapefile_with_rr_data(shapefile, da_rr_54034).reset_index(drop=True)

        if intersecting_shapes.empty:
            continue
        
        # rasterize storm relative risk
        rr_raster_54034 = to_raster(
            ds=da_rr_54034,
            no_data_value=np.nan,
            lat_col="y",
            lon_col="x",
            crs="ESRI:54034",
        )

        # subset to affected area
        rr_raster_54034_subset = subset_affected_area(
            rr_raster=rr_raster_54034,     
            threshold=0.0,
        )

        # read in full population data using bouds
        pop_raster_full = pop_dict[year]

        # subset full pop_raster to match rr_raster_54034_subset
        xmin, ymin, xmax, ymax = rr_raster_54034_subset.bounds
        bbox_geom = box(xmin, ymin, xmax, ymax)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=rr_raster_54034_subset.crs)

        # clip & mask population raster using the bbox polygon
        pop_raster_subset = (
            pop_raster_full
            .clip(bbox_gdf)
        )

        # downscale to match population
        clim_arr = (
            rr_raster_54034_subset
            .resample_to(pop_raster_subset, "nearest")
            .astype(np.float32)
        )

        # calculate pafs
        paf_raster_df = compute_paf_raster_per_admin(
            intersecting_shapes=intersecting_shapes, 
            rr_raster=clim_arr, 
            pop_raster=pop_raster_full,
            location_id_col="location_id",
            days_impact=20,
        )

        # add to current paf df
        paf_df_list.append(paf_raster_df)

        # ---- Explicit cleanup ----
        del storm_rr
        del da_rr
        del da_rr_54034
        del rr_raster_54034
        del rr_raster_54034_subset
        del pop_raster_subset
        del clim_arr

        gc.collect()


        
    # concatenate all paf dfs
    paf_df = pd.concat(paf_df_list, ignore_index=True)

    return paf_df

def save_paf(
        source_id: str,
        variant_label: str,
        experiment_id: str,
        batch_year: str,
        basin: str,
        draw: int,
        storm_index: int | None,
        paf_df: pd.DataFrame,
    ):
    # test save root
    save_root = Path("/mnt/share/scratch/users/mfiking/climada_pafs")

    paf_dir = save_root / source_id / variant_label / experiment_id / batch_year / basin 
    paf_dir.mkdir(parents=True, exist_ok=True)

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    paf_path = paf_dir / f"paf{draw_text}_s{storm_index}.parquet"

    paf_df.to_parquet(paf_path, index=False)
    os.chmod(paf_path, 0o775)
    print(f"✅ Saved PAF to {paf_path}")

def main(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    storm_index: int | None,
    sample_name: str,
    relative_risk: str,
):
    # prepare TCTracks from Zarr files
    tc_tracks = prepare_zarr_files(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        storm_index=storm_index,
    )

    # generate basin centroids
    centroids = generate_basin_centroids(basin, res=RESOLUTION)

    # generate hazard object
    haz = generate_hazard_per_track(tc_tracks, centroids)

    # generate per-storm wind speed - NOT USED - FOR EXPOSURE CALCULATION ONLY
    storm_list_speed = generate_speed_per_storm(haz, centroids, tc_tracks)

    # generate per-storm intensity
    storm_list_intensity = generate_intensity_per_storm(haz, centroids, tc_tracks)

    # generate pafs
    paf_df = generate_pafs(storm_list_intensity, sample_name, basin, relative_risk)

    # save paf
    save_paf(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        storm_index=storm_index,
        paf_df=paf_df,
    )

    return paf_df




main(
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
    batch_year=batch_year,
    basin=basin,
    draw=draw,
    storm_index=storm_index,
    sample_name=sample_name,
    relative_risk=relative_risk,
)
