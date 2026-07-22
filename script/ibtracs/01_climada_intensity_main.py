"""
Stage 1 (IBTrACS): per-storm intensity + exposure-hours rasters from
historical IBTrACS tracks for one (year, basin). Mirrors the CMIP6
stage-1 worker (`script/climada/01_climada_intensity_main.py`) but with
IBTrACS-shaped inputs and NetCDF outputs (one file per storm).
"""

import json
import re
import os
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import rasterra as rt  # type: ignore

from climada.hazard import TCTracks, TropCyclone, Centroids
from shapely.geometry import box  # type: ignore
from affine import Affine  # type: ignore
from rasterio.errors import WindowError

import logging

logging.getLogger("climada").setLevel(logging.WARNING)


parser = argparse.ArgumentParser(description="Run IBTracs historical storm data processing code")

# Define arguments
parser.add_argument("--year", type=int, required=True, help="Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")

# Parse arguments
args = parser.parse_args()
year = args.year
basin = args.basin


# Constants
ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage1/")
LOG_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage1_logs/")

RESOLUTION = 0.1  # degrees
GDF_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84.parquet")
SHP_ROOT_NORMALIZED = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29')

############################################
#              Helper Functions            #
############################################
def ensure_min_grid(da: xr.DataArray, buffer_deg: float = 0.1) -> xr.DataArray:
    if da.lat.size == 1:
        c = float(da.lat.values[0])
        da = da.reindex(lat=[c-buffer_deg, c, c+buffer_deg], fill_value=0)

    if da.lon.size == 1:
        c = float(da.lon.values[0])
        da = da.reindex(lon=[c-buffer_deg, c, c+buffer_deg], fill_value=0)

    return da

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


def generate_storm_specific_basin(storm_event, res=0.1, buffer_deg=5.0):
    """
    Generate centroids for a specific storm based on its track extent.
    """

    xmin = storm_event.lon.min().values
    xmax = storm_event.lon.max().values
    ymin = storm_event.lat.min().values
    ymax = storm_event.lat.max().values

    # Apply buffer    
    xmin = xmin - buffer_deg
    xmax = xmax + buffer_deg
    ymin = ymin - buffer_deg
    ymax = ymax + buffer_deg
    
    centroids = Centroids.from_pnt_bounds(
        (xmin, ymin, xmax, ymax),
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

def generate_speed_single_storm(
    haz: TropCyclone,
    centroids: Centroids,
    tc_tracks: TCTracks,
    basin: str,
) -> xr.DataArray | None:
    """
    Generate per-storm wind speed DataArray for a single storm.

    Parameters
    ----------
    haz : TropCyclone
        Hazard object containing windfields for all storms.
    centroids : Centroids
        Grid centroids for the basin.
    tc_tracks : TCTracks
        Single-storm TCTracks object (length 1).

    Returns
    -------
    xr.DataArray | None
        Per-pixel wind speed DataArray for the storm, or None if skipped.
    """
    storm_event = tc_tracks.data[0]  # single storm

    # canonical grid
    lon_vals = np.sort(np.unique(centroids.coord[:, 1]))
    lat_vals = np.sort(np.unique(centroids.coord[:, 0]))[::-1]  # descending
    n_lat, n_lon = len(lat_vals), len(lon_vals)
    assert centroids.coord.shape[0] == n_lat * n_lon

    # locate the storm index in haz.windfields
    storm_idx = 0  # single-storm TCTracks always has one entry

    try:
        wf = haz.windfields[storm_idx].toarray()
    except Exception as e:
        print(f"⚠️ Could not read windfield for {storm_event.name}: {e}")
        return None

    times = storm_event.time
    n_time = len(times)

    try:
        wf_reshaped = wf.reshape(n_time, n_lat, n_lon, 2)
    except ValueError:
        print(f"⚠️ Skipping storm {storm_event.name}: shape mismatch")
        return None

    da = xr.DataArray(
        wf_reshaped,
        coords={
            "time": times,
            "lat": lat_vals,
            "lon": lon_vals,
            "dir": ["u", "v"],
        },
        dims=["time", "lat", "lon", "dir"],
        name=f"{storm_event.name}_windfields",
    )

    # compute wind speed
    da_speed = np.sqrt(da.isel(dir=0) ** 2 + da.isel(dir=1) ** 2)

    # time_step handling
    if "time_step" in storm_event:
        time_step = storm_event["time_step"].values.astype("float32")
    else:
        dt = np.diff(times.values) / np.timedelta64(1, "h")
        time_step = np.append(dt, dt[-1]).astype("float32")

    da_speed = da_speed.assign_coords(time_step=("time", time_step))

    start_date = pd.to_datetime(times.values[0])
    end_date = pd.to_datetime(times.values[-1])

    # metadata
    da_speed.attrs.update({
        "description": f"Storm {storm_event.name} wind speed",
        "units": "m/s",
        "storm_name": storm_event.name,
        "storm_id": storm_event.sid,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "basin": basin,
        "category": str(storm_event.attrs.get("category", "")),
        "time_step_units": "hours",
    })

    return da_speed
######################################
#    Yearly Exposure Functions       #
######################################

def compute_exposure_single_storm(
    storm_da: xr.DataArray,
    basin: str,
    wind_threshold: float = 17.0,
) -> xr.DataArray | None:
    """
    Compute per-storm, per-year exposure hours at pixel level for a single storm.
    Returns None if exposure is empty or fails land filter.
    """

    if "time" not in storm_da.coords or "time_step" not in storm_da.coords:
        return None

    exposure = xr.where(storm_da > wind_threshold, storm_da["time_step"], 0.0)

    # group by year
    time_index = pd.DatetimeIndex(storm_da.time.values)
    year_groups = time_index.to_period("Y").to_timestamp()
    group_da = xr.DataArray(year_groups, dims="time", coords={"time": storm_da.time}, name="year")
    yearly_exposure = exposure.groupby(group_da).sum(dim="time")

    if "year" in yearly_exposure.dims:
        yearly_exposure = yearly_exposure.rename({"year": "time"})

    yearly_exposure = yearly_exposure.assign_coords(time=np.array(yearly_exposure.time.values, dtype="datetime64[ns]"))
    yearly_exposure = yearly_exposure.astype("float32")

    # metadata
    yearly_exposure.name = "exposure_hours"
    yearly_exposure.attrs.update({
        "storm_name": storm_da.attrs.get("storm_name"),
        "storm_id": storm_da.attrs.get("storm_id"),

        "start_date": pd.to_datetime(storm_da.attrs.get("start_date")).isoformat(),
        "end_date": pd.to_datetime(storm_da.attrs.get("end_date")).isoformat(),

        # FIXED
        "basin": np.array(storm_da.attrs.get("basin", basin), ndmin=1).astype(str),
        "category": str(storm_da.attrs.get("category", "")),

        "description": f"Per-storm yearly exposure hours per pixel where wind speed > {wind_threshold} m/s",
        "definition": "Exposure hours are the sum of timestep durations with wind speed above threshold",
        "units": "hours",
        "aggregation": "yearly",
        "wind_threshold_m_s": float(wind_threshold),
    })

    yearly_exposure.attrs.pop("time_step", None)

    # collapse single-year storms
    if yearly_exposure.sizes.get("time", 0) == 1:
        yearly_exposure = yearly_exposure.isel(time=0, drop=True)

    # NA basin normalization
    if basin == "NA":
        yearly_exposure = yearly_exposure.assign_coords(lon=((yearly_exposure.lon + 180) % 360) - 180).sortby("lon")

    # load land polygons
    land_gdf = (
        load_land_polygons_for_storm_normalized(yearly_exposure, SHP_ROOT_NORMALIZED, buffer=5.0)
        if basin == "NA"
        else load_land_polygons_for_storm(yearly_exposure, GDF_PATH, buffer=5.0)
    )
    yearly_exposure = clip_raster_to_land(yearly_exposure, land_gdf)

    if yearly_exposure.isnull().all():
        return None

    del land_gdf, exposure

    return yearly_exposure

######################################
#    Per Storm Intensity Functions   #
######################################

def generate_intensity_single_storm(
    haz: TropCyclone,
    centroids: Centroids,
    tc_tracks: TCTracks,
    basin: str,
    wind_threshold: float = 17.0,
) -> xr.DataArray | None:
    """
    Generate per-storm intensity for a single TCTracks object (1 storm only).
    Returns a DataArray for the storm and the list of valid storm IDs (1 or empty).
    """
    # --- Grid definition ---
    lon_vals = np.sort(np.unique(centroids.coord[:, 1]))
    lat_vals = np.sort(np.unique(centroids.coord[:, 0]))[::-1]  # descending
    n_lat = len(lat_vals)
    n_lon = len(lon_vals)
    assert centroids.coord.shape[0] == n_lat * n_lon

    # load intensity array once
    intensity_all = haz.intensity.toarray()  # shape: (n_storms, n_centroids)
    event = tc_tracks.data[0]

    storm_name = event.name
    storm_id = event.sid
    storm_category = getattr(event, "category", None)
    times = event.time
    start_date = pd.to_datetime(times.values[0])
    end_date = pd.to_datetime(times.values[-1])

    try:
        i = 0  # single storm
        intensity_flat = intensity_all[i, :]
    except Exception as e:
        print(f"⚠️ Could not read intensity for {storm_name}: {e}")
        return None

    if intensity_flat.size != n_lat * n_lon:
        print(f"⚠️ Skipping {storm_name}: grid mismatch")
        return None

    if np.nanmax(intensity_flat) < wind_threshold:
        return None

    intensity_2d = intensity_flat.reshape(n_lat, n_lon)
    da = xr.DataArray(
        intensity_2d,
        coords={"lat": lat_vals, "lon": lon_vals},
        dims=["lat", "lon"],
        name=storm_name,
    )

    # Normalize NA basin
    if basin == "NA":
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")

    # load land polygons
    land_gdf = (
        load_land_polygons_for_storm_normalized(da, SHP_ROOT_NORMALIZED, buffer=5.0)
        if basin == "NA"
        else load_land_polygons_for_storm(da, GDF_PATH, buffer=5.0)
    )

    # landfall check
    if not check_storm_landfall(da, land_gdf, wind_threshold):
        return None

    da.attrs.update({
        "description": "Per-storm pixel-level maximum wind speed",
        "units": "m/s",
        "storm_name": storm_name,
        "storm_id": storm_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "basin": basin,
        "category": storm_category,
        "definition": "Maximum wind speed experienced at each pixel during storm lifetime",
    })

    # clip to land
    da = clip_raster_to_land(da, land_gdf)
    if da.isnull().all():
        return None

    return da



#########################################
#     Check Landfall Functions          #
#########################################

def load_land_polygons_for_storm(
    storm_da,
    shapefile_gdf_path: str,
    buffer: float = 2.0,
) -> gpd.GeoDataFrame:
    """
    Load land polygons intersecting storm bounding box (0–360 longitude space).
    """

    min_lon = float(storm_da["lon"].min()) - buffer
    max_lon = float(storm_da["lon"].max()) + buffer
    min_lat = float(storm_da["lat"].min()) - buffer
    max_lat = float(storm_da["lat"].max()) + buffer

    gdf = gpd.read_parquet(shapefile_gdf_path)

    # --------------------------------------------------
    # Handle longitude wraparound (0–360 domain)
    # --------------------------------------------------
    if min_lon < 0 or max_lon > 360:
        # wrapped bbox → split into two
        boxes = []

        if min_lon < 0:
            boxes.append(box(0, min_lat, max_lon, max_lat))
            boxes.append(box(360 + min_lon, min_lat, 360, max_lat))

        elif max_lon > 360:
            boxes.append(box(min_lon, min_lat, 360, max_lat))
            boxes.append(box(0, min_lat, max_lon - 360, max_lat))

        bbox_geom = gpd.GeoSeries(boxes, crs=gdf.crs).union_all()

    else:
        bbox_geom = box(min_lon, min_lat, max_lon, max_lat)

    gdf_subset = gdf[gdf.intersects(bbox_geom)].copy()
    gdf_subset["geometry"] = gdf_subset.geometry.intersection(bbox_geom)

    return gdf_subset

def load_land_polygons_for_storm_normalized(
    storm_da,
    shapefile_root: Path,
    buffer: float = 2.0,
) -> gpd.GeoDataFrame:
    """
    Load land polygons intersecting storm bounding box (-180 to 180 longitude space).
    """
    admin_level = 0
    simplified_suffix = ''
    # --- Bounding box ---
    min_lon = float(storm_da["lon"].min()) - buffer
    max_lon = float(storm_da["lon"].max()) + buffer
    min_lat = float(storm_da["lat"].min()) - buffer
    max_lat = float(storm_da["lat"].max()) + buffer

    # --- Load shapefile ---
    shp_path = shapefile_root / f"lbd_standard_admin_{admin_level}{simplified_suffix}.shp"
    gdf = gpd.read_file(shp_path)

    # --- Simple bbox (no wraparound needed in -180 to 180) ---
    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)

    # --- Spatial filter ---
    gdf_subset = gdf[gdf.intersects(bbox_geom)].copy()
    gdf_subset["geometry"] = gdf_subset.geometry.intersection(bbox_geom)

    return gdf_subset

def check_storm_landfall(
    storm_da: xr.DataArray,
    land_gdf: "gpd.GeoDataFrame",
    wind_threshold: float = 17.0,
) -> bool:
    """
    Determine if a storm makes landfall.

    Returns True only if storm has winds > wind_threshold
    AND those winds intersect land.

    Uses RasterArray.clip().mask() to handle small islands robustly.
    """

    # --- 0. Skip globally weak storms ---
    if float(storm_da.max().values) < wind_threshold:
        return False

    # --- 1. Ensure minimal grid (single-pixel storms) ---
    storm_da = ensure_min_grid(storm_da)

    if storm_da.lat.size < 2 or storm_da.lon.size < 2:
        return False

    # --- 2. Convert to RasterArray ---
    raster = to_raster(
        ds=storm_da,
        no_data_value=np.nan,
        lat_col="lat",
        lon_col="lon",
        crs="EPSG:4326",
    )

    xmin, xmax, ymin, ymax = raster.bounds

    raster_bbox = box(xmin, ymin, xmax, ymax)

    # --- 3. Clip and mask to land ---
    geom = land_gdf.geometry.union_all()  # Polygon or MultiPolygon

    # --- KEY FIX ---
    if not raster_bbox.intersects(geom):
        return False  # storm is fully over ocean

    # 1. Clip first
    try:
        raster_clipped = raster.clip(geom)
    except WindowError:
        return False  # no overlap in pixel space

    # 2. Check if any valid pixels
    if raster_clipped._ndarray.size == 0:
        return False

    # 3. Mask only if needed
    try:
        raster_land = raster_clipped.mask(geom, all_touched=True)
    except WindowError:
        return False

    if raster_land._ndarray.size == 0:
        return False


    # --- 4. Apply wind threshold ---
    data = raster_land._ndarray
    strong_wind_mask = np.isfinite(data) & (data > wind_threshold)
    # --- 5. Check if any pixel survives ---
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


def subset_affected_area(
    rr_raster: rt.RasterArray,
    threshold: float = 0.0,
    buffer_pixels: int = 2,
) -> rt.RasterArray:
    """
    Subset a RasterArray to the minimal bounding box
    where RR > threshold, with optional pixel buffer.

    Parameters
    ----------
    rr_raster : RasterArray
        Storm relative risk raster.
    threshold : float
        Threshold defining affected pixels.
    buffer_pixels : int
        Number of pixels to expand the bounding box in all directions.

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

    # Pixel → coordinate conversion with buffer
    xmin = c + (cols.min() - buffer_pixels) * a
    xmax = c + (cols.max() + 1 + buffer_pixels) * a
    ymax = f + (rows.min() - buffer_pixels) * e
    ymin = f + (rows.max() + 1 + buffer_pixels) * e

    # Build geometry
    geom = box(xmin, ymin, xmax, ymax)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs=rr_raster.crs)

    # Native rasterra clip
    return rr_raster.clip(gdf)


def clip_raster_to_land(
    da: xr.DataArray,
    land_gdf: "gpd.GeoDataFrame",
) -> xr.DataArray:

    # --- 0️⃣ Ensure minimum grid (only if degenerate) ---
    if da.sizes.get("lat", 0) < 2 or da.sizes.get("lon", 0) < 2:
        da = ensure_min_grid(da, buffer_deg=0.1)
        # ensure padding is NaN, not 0
        da = da.where(da != 0, np.nan)

    # 1️⃣ Union land polygons
    geom = land_gdf.geometry.union_all()

    # 2️⃣ Convert to RasterArray
    raster = to_raster(
        ds=da,
        no_data_value=np.nan,
        lat_col="lat",
        lon_col="lon",
        crs="EPSG:4326",
    )

    # 3️⃣ Clip + mask (guard against empty intersection)
    try:
        raster_land = raster.clip(geom).mask(geom, all_touched=True)
    except Exception:
        # return empty raster with same structure
        return da * np.nan
    
    # subset to affected area
    try:
        raster_land = subset_affected_area(raster_land, threshold=0.0)
    except ValueError:
        # No affected pixels → return empty raster with same structure
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
def normalize_basin(basin):
    if isinstance(basin, (list, tuple)):
        return basin[0]
    return basin


def save_storm_intensity(
    da: xr.DataArray,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    save_root: Path = SAVE_ROOT,
):
    """
    Save a single storm intensity DataArray as a NetCDF using storm_id.
    Metadata is stored at dataset level (recommended schema).
    """

    out_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / str(year)
        / basin
        / "intensity"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    da = da.copy()
    da.name = "intensity"

    if da.dtype != "float32":
        da = da.astype("float32")

    # --- normalize basin ---
    basin = normalize_basin(da.attrs.get("basin", basin))

    # --- required metadata ---
    storm_id = da.attrs["storm_id"]

    # --- build dataset-level metadata ---
    attrs = dict(da.attrs)
    attrs["basin"] = basin

    ds = da.to_dataset(name="intensity")
    ds.attrs = attrs

    out_file = out_dir / f"{storm_id}.nc"

    # --- safe overwrite ---
    if out_file.exists():
        out_file.unlink()

    # --- save ---
    ds.to_netcdf(out_file, mode="w")

    os.chmod(out_file, 0o775)
    os.chmod(out_dir, 0o775)

def save_storm_exposure(
    da: xr.DataArray,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    save_root: Path = SAVE_ROOT,
):
    """
    Save a single storm exposure DataArray as a NetCDF using storm_id.
    Metadata is stored at dataset level (recommended schema).
    """

    out_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / str(year)
        / basin
        / "exposure_hours"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    da = da.copy()
    da.name = "exposure_hours"

    if da.dtype != "float32":
        da = da.astype("float32")

    # --- normalize basin ---
    basin = normalize_basin(da.attrs.get("basin", basin))

    storm_id = da.attrs["storm_id"]

    attrs = dict(da.attrs)
    attrs["basin"] = basin

    ds = da.to_dataset(name="exposure_hours")
    ds.attrs = attrs

    out_file = out_dir / f"{storm_id}.nc"

    # --- safe overwrite ---
    if out_file.exists():
        out_file.unlink()

    ds.to_netcdf(out_file, mode="w")

    os.chmod(out_file, 0o775)
    os.chmod(out_dir, 0o775)


def prepare_track_for_climada(ds_track: xr.Dataset):

    return TCTracks(data=[ds_track])
def check_if_storm_nc_complete(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    storm_id: str | int,
    parameter: str = "intensity",  # or "exposure_hours"
    save_root: Path = SAVE_ROOT,
) -> bool:
    """
    Return True if a per-storm NetCDF file exists and appears valid.
    """
    out_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / str(year)
        / basin
        / parameter
    )
    file_path = out_dir / f"{storm_id}.nc"

    if not file_path.exists() or file_path.stat().st_size == 0:
        return False

    # Optional: lightweight xarray validation
    try:
        with xr.open_dataset(file_path, decode_timedelta=False) as ds:
            if not ds.dims:
                return False
    except Exception:
        return False

    return True

def log_storm_id_json(
    storm_id: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    parameter: str,
    status: str,  # <-- IMPORTANT (completed / skipped / failed)
    message: str | None = None,
    log_root: Path = LOG_ROOT,
):
    """
    Log a processed storm ID to a JSON file for tracking.
    Lightweight per-storm audit log.
    """

    log_dir = (
        log_root
        / source_id
        / variant_label
        / experiment_id
        / str(year)
        / basin
        / parameter
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{storm_id}.json"

    payload = {
        "storm_id": storm_id,
        "source_id": source_id,
        "variant_label": variant_label,
        "experiment_id": experiment_id,
        "year": str(year),
        "basin": basin,
        "parameter": parameter,
        "status": status,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # overwrite-safe write
    tmp_file = log_file.with_suffix(".tmp")

    with open(tmp_file, "w") as f:
        json.dump(payload, f, indent=2)

    os.replace(tmp_file, log_file)

    os.chmod(log_file, 0o775)
    os.chmod(log_dir, 0o775)

def is_storm_completed(
    storm_id: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    log_root: Path,
) -> bool:
    """
    Check if a storm has already been fully processed.
    """

    log_file = (
        log_root
        / source_id
        / variant_label
        / experiment_id
        / str(year)
        / basin
        / "intensity_and_exposure"
        / f"{storm_id}.json"
    )

    if not log_file.exists():
        return False

    try:
        with open(log_file, "r") as f:
            data = json.load(f)

        return data.get("status") == "completed"

    except Exception:
        return False

def save_storm_extent(
    storm_df_extent: pd.DataFrame,
    storm_id: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    save_root: Path = SAVE_ROOT,
):
    
    # save storm extent parquet
    storm_extent_dir = (
        save_root /
        source_id /
        variant_label /
        experiment_id /
        str(year) /
        basin /
        "storm_extent"
    )
    file_name = f"{storm_id}.parquet"
    extent_out_file = storm_extent_dir / file_name

    extent_out_file.parent.mkdir(parents=True, exist_ok=True)
    storm_df_extent.to_parquet(extent_out_file, index=False)
    os.chmod(extent_out_file, 0o775)

def _task_complete_marker_path(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    save_root: Path = SAVE_ROOT,
) -> Path:
    """Sentinel marker file. Presence = the (year, basin) task finished
    main() successfully (including the no-storms-this-year case). Used by
    the launcher to skip re-running already-completed tasks."""
    return (
        save_root / source_id / variant_label / experiment_id
        / str(year) / basin / "_task_complete.flag"
    )


def write_task_complete_marker(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    save_root: Path = SAVE_ROOT,
):
    """Touch the sentinel marker at the end of a successful main()."""
    marker = _task_complete_marker_path(
        source_id=source_id, variant_label=variant_label,
        experiment_id=experiment_id, year=year, basin=basin,
        save_root=save_root,
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now(timezone.utc).isoformat())
    os.chmod(marker, 0o775)
    os.chmod(marker.parent, 0o775)


def save_full_storm_extent_summary(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    save_root: Path,
):
    storm_extent_dir = (
        save_root /
        source_id /
        variant_label /
        experiment_id /
        str(year) /
        basin /
        "storm_extent"
    )
        
    summary_list = []

    # if no storm extents were saved, skip summary generation
    if not storm_extent_dir.exists():
        print(f"No storm extent data found for {basin} in {year}, skipping summary generation.")
        return
    
    for file in storm_extent_dir.glob("*.parquet"):
        df = pd.read_parquet(file)
        summary_list.append(df)
    if summary_list:
        summary_df = pd.concat(summary_list, ignore_index=True)
        summary_out_file = storm_extent_dir.parent / "storm_extent_summary.csv"
        summary_df.to_csv(summary_out_file, index=False)
        os.chmod(summary_out_file, 0o775)
############################################
#              Main                        #
############################################

def process_single_storm(
    storm_event,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str,
    basin: str,
    save_root: Path,
    log_root: Path,
) -> dict | None:
    """
    Process one IBTrACS storm: validate the basin tag, short-circuit on
    existing outputs, compute intensity + exposure + wind speed, save the
    NetCDFs, and log completion. Returns the extent-row dict on success,
    or None when the storm is deliberately skipped (basin mismatch,
    already complete, or no valid intensity/speed/exposure). Errors
    propagate to the caller, which catches and continues to the next storm.
    """
    storm_id = storm_event.sid

    # get first time step basin from storm event
    first_basin = storm_event.sel(time=storm_event.time[0])["basin"].values.item()
    print(f"First time step basin: {first_basin}")

    if first_basin != basin:
        print(f"⚠️ Skipping {storm_id}, first time step basin {first_basin} does not match target basin {basin}")
        return None

    # 🔥 EARLY EXIT (fastest possible skip)
    if is_storm_completed(
        storm_id=storm_id,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        year=year,
        basin=basin,
        log_root=log_root,
    ):
        print(f"✅ Skipping {storm_id} (already completed)")
        return None

    # Check if both intensity and exposure are already saved
    intensity_done = check_if_storm_nc_complete(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        year=year,
        basin=basin,
        storm_id=storm_id,
        parameter="intensity",
        save_root=save_root,
    )
    exposure_done = check_if_storm_nc_complete(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        year=year,
        basin=basin,
        storm_id=storm_id,
        parameter="exposure_hours",
        save_root=save_root,
    )
    if intensity_done and exposure_done:
        print(f"✅ Skipping {storm_id}, intensity and exposure already processed")
        return None

    centroids = generate_storm_specific_basin(storm_event, res=RESOLUTION, buffer_deg=5.0)
    tc_tracks = prepare_track_for_climada(storm_event)
    haz = generate_hazard_per_track(tc_tracks, centroids)

    storm_intensity = generate_intensity_single_storm(
        haz=haz,
        centroids=centroids,
        tc_tracks=tc_tracks,
        basin=basin,
        wind_threshold=17.0,
    )
    if storm_intensity is None:
        print(f"⚠️  Skipping {storm_id}, no valid intensity")
        log_storm_id_json(
            storm_id=storm_id, source_id=source_id, variant_label=variant_label,
            experiment_id=experiment_id, year=year, basin=basin,
            parameter="intensity", status="skipped",
            message="No valid intensity", log_root=log_root,
        )
        return None
    storm_intensity = storm_intensity.where(storm_intensity > 0, np.nan)

    storm_speed = generate_speed_single_storm(
        haz=haz,
        centroids=centroids,
        tc_tracks=tc_tracks,
        basin=basin,
    )
    if storm_speed is None:
        print(f"⚠️  Skipping {storm_id}, no valid wind speed")
        log_storm_id_json(
            storm_id=storm_id, source_id=source_id, variant_label=variant_label,
            experiment_id=experiment_id, year=year, basin=basin,
            parameter="wind_speed", status="skipped",
            message="No valid wind speed", log_root=log_root,
        )
        return None

    storm_exposure = compute_exposure_single_storm(
        storm_da=storm_speed,
        basin=basin,
        wind_threshold=17.0,
    )
    if storm_exposure is None:
        print(f"⚠️  Skipping {storm_id}, no valid exposure")
        log_storm_id_json(
            storm_id=storm_id, source_id=source_id, variant_label=variant_label,
            experiment_id=experiment_id, year=year, basin=basin,
            parameter="exposure_hours", status="skipped",
            message="No valid exposure", log_root=log_root,
        )
        return None
    storm_exposure = storm_exposure.where(storm_exposure > 0, np.nan)

    save_storm_intensity(
        da=storm_intensity,
        source_id=source_id, variant_label=variant_label,
        experiment_id=experiment_id, year=year, basin=basin,
        save_root=save_root,
    )
    save_storm_exposure(
        da=storm_exposure,
        source_id=source_id, variant_label=variant_label,
        experiment_id=experiment_id, year=year, basin=basin,
        save_root=save_root,
    )

    log_storm_id_json(
        storm_id=storm_id, source_id=source_id, variant_label=variant_label,
        experiment_id=experiment_id, year=year, basin=basin,
        parameter="intensity_and_exposure",
        status="completed",
        message="Successfully processed intensity and exposure",
        log_root=log_root,
    )

    extent_row = {
        "storm_id": storm_id,
        "storm_name": tc_tracks.data[0].name,
        "start_date": tc_tracks.data[0].time[0].values.astype("datetime64[D]"),
        "end_date": tc_tracks.data[0].time[-1].values.astype("datetime64[D]"),
        "lon_min": float(storm_intensity.lon.min().values),
        "lon_max": float(storm_intensity.lon.max().values),
        "lat_min": float(storm_intensity.lat.min().values),
        "lat_max": float(storm_intensity.lat.max().values),
    }
    return extent_row


def main(
    year: int | str,
    basin: str,
    save_root: Path = SAVE_ROOT,
    log_root: Path = LOG_ROOT,
):

    # metadata for saving
    source_id = "ibtracs"
    variant_label = "official"
    experiment_id = "historical"


    tracks = TCTracks.from_ibtracs_netcdf(provider="official", basin=basin, year_range=(year, year))
    year = str(year)

    if not tracks.data:
        print(f"No storms found for {basin} in {year}")
        write_task_complete_marker(
            source_id=source_id, variant_label=variant_label,
            experiment_id=experiment_id, year=year, basin=basin,
            save_root=save_root,
        )
        return

    print(f"Generated hazard for {len(tracks.data)} storms in {basin} basin for year {year}.")

    extent_rows = []
    for i, storm_event in enumerate(tracks.data):
        storm_id = storm_event.sid
        print(f"Processing storm {i+1}/{len(tracks.data)}: {storm_event.name} (ID: {storm_id})")
        try:
            extent_row = process_single_storm(
                storm_event=storm_event,
                source_id=source_id, variant_label=variant_label,
                experiment_id=experiment_id, year=year, basin=basin,
                save_root=save_root, log_root=log_root,
            )
        except Exception as e:
            print(f"❌ Storm {storm_id} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            try:
                log_storm_id_json(
                    storm_id=storm_id, source_id=source_id,
                    variant_label=variant_label,
                    experiment_id=experiment_id, year=year, basin=basin,
                    parameter="intensity_and_exposure",
                    status="failed",
                    message=f"{type(e).__name__}: {e}",
                    log_root=log_root,
                )
            except Exception:
                pass
            continue

        if extent_row is None:
            continue

        extent_rows.append(extent_row)
        save_storm_extent(
            storm_df_extent=pd.DataFrame(extent_rows),
            storm_id=extent_row["storm_id"],
            source_id=source_id, variant_label=variant_label,
            experiment_id=experiment_id, year=year, basin=basin,
            save_root=save_root,
        )

    # save storm extent summary
    save_full_storm_extent_summary(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        year=year,
        basin=basin,
        save_root=save_root,
    )

    write_task_complete_marker(
        source_id=source_id, variant_label=variant_label,
        experiment_id=experiment_id, year=year, basin=basin,
        save_root=save_root,
    )

main(
    year=year,
    basin=basin,
    save_root=SAVE_ROOT,
    log_root=LOG_ROOT,
)