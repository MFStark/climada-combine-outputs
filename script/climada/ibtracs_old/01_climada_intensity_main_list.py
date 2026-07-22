from pathlib import Path
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
from climada.hazard import TCTracks, TropCyclone, Centroids
import gc
import re
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import os
from shapely.geometry import box  # type: ignore
import argparse
from affine import Affine  # type: ignore
from rasterio.errors import WindowError
import rasterra as rt  # type: ignore
import stat

import logging

logging.getLogger("climada").setLevel(logging.WARNING)


parser = argparse.ArgumentParser(description="Run IBTracks historical storm data processing code")

# Define arguments
parser.add_argument("--year", type=int, required=True, help="Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")

# Parse arguments
args = parser.parse_args()
year = args.year
basin = args.basin


# Constants
ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracks_stage1_v2/")
RESOLUTION = 0.1  # degrees
GDF_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84.parquet")
SHP_ROOT_NORMALIZED = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29')

############################################
#              Helper Functions            #
############################################
def chmod_recursive(path: Path, mode: int = 0o775):
    """
    Recursively set permissions for a directory and all its files/subdirectories.
    """
    path = Path(path)
    if path.is_dir():
        os.chmod(path, mode)
        for child in path.iterdir():
            chmod_recursive(child, mode)
    elif path.exists():
        os.chmod(path, mode)

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
    valid_storm_ids: list[str],
) -> list[xr.DataArray]:
    """
    Generate per-storm wind speed DataArrays for all storms.
    """

    # --- Grid definition (canonical, no flipping) ---
    lon_vals = np.sort(np.unique(centroids.coord[:, 1]))
    lat_vals = np.sort(np.unique(centroids.coord[:, 0]))

    n_lat = len(lat_vals)
    n_lon = len(lon_vals)

    # validate grid
    assert centroids.coord.shape[0] == n_lat * n_lon

    storm_list_speed = []

    for i, event in enumerate(tc_tracks.data):

        storm_name = event.name
        storm_id = event.sid
        storm_basin = getattr(event, "basin", None)
        storm_category = getattr(event, "category", None)

        # filter storms by valid storm ids
        if storm_id not in valid_storm_ids:
            continue

        times = event.time
        n_time = len(times)

        # --- Dates (robust) ---
        storm_start_date = pd.to_datetime(times.values[0])
        storm_end_date = pd.to_datetime(times.values[-1])

        # --- Windfield ---
        wf = haz.windfields[i].toarray()

        try:
            wf_reshaped = wf.reshape(n_time, n_lat, n_lon, 2)
        except ValueError:
            print(f"⚠️ Skipping storm {storm_name}: shape mismatch")
            continue

        # --- Create DataArray ---
        da = xr.DataArray(
            wf_reshaped,
            coords={
                "time": times,
                "lat": lat_vals,
                "lon": lon_vals,
                "dir": ["u", "v"],
            },
            dims=["time", "lat", "lon", "dir"],
            name=f"{storm_name}_windfields",
        )

        # --- Compute wind speed ---
        da_speed = np.sqrt(da.isel(dir=0) ** 2 + da.isel(dir=1) ** 2)

        # --------------------------------------------------
        # Timestep handling (correct for IBTrACS)
        # --------------------------------------------------
        if "time_step" in event:
            time_step = event["time_step"].values.astype("float32")
        else:
            dt = np.diff(times.values) / np.timedelta64(1, "h")
            time_step = np.append(dt, dt[-1]).astype("float32")

        assert len(time_step) == n_time

        da_speed = da_speed.assign_coords(
            time_step=("time", time_step)
        )

        # --- Metadata ---
        da_speed.name = storm_name
        da_speed.attrs.update({
            "description": f"Storm {storm_name} wind speed",
            "units": "m/s",
            "storm_name": storm_name,
            "storm_id": storm_id,
            "start_date": storm_start_date,
            "end_date": storm_end_date,
            "basin": storm_basin,
            "category": storm_category,
            "time_step_units": "hours",
        })

        storm_list_speed.append(da_speed)

        # --- Free memory ---
        del wf, wf_reshaped, da, da_speed

    return storm_list_speed

######################################
#    Yearly Exposure Functions       #
######################################

def compute_yearly_exposure_per_storm_list(
    storm_list: list[xr.DataArray],
    basin: str,
    wind_threshold: float = 17.0,
) -> tuple[list[xr.DataArray], list[str]]:
    """
    Compute per-storm, per-year exposure hours at pixel level for a list of storms.
    Uses variable timestep weighting (IBTrACS-compatible).
    """

    exposure_list = []
    valid_exposure_ids = []

    for storm_da in storm_list:

        storm_name = storm_da.name

        # --- Safety checks ---
        if "time" not in storm_da.coords:
            raise ValueError(f"Storm {storm_name} missing 'time' coordinate")

        if "time_step" not in storm_da.coords:
            raise ValueError(f"Storm {storm_name} missing 'time_step' coordinate")

        # --------------------------------------------------------
        # 1. Threshold → weighted exposure (core fix)
        # --------------------------------------------------------
        exposure = xr.where(
            storm_da > wind_threshold,
            storm_da["time_step"],
            0.0
        )

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
                "Exposure hours are computed as the sum of timestep durations "
                "with wind speed above the threshold."
            ),
            "units": "hours",
            "aggregation": "yearly",
            "wind_threshold_m_s": wind_threshold,
        })

        # remove timestep attr if propagated
        yearly_exposure.attrs.pop("time_step", None)

        # --------------------------------------------------------
        # 4. Collapse single-year storms
        # --------------------------------------------------------
        if yearly_exposure.sizes.get("time", 0) == 1:
            yearly_exposure = yearly_exposure.isel(time=0, drop=True)

        # if basin is NA normalize
        if basin == "NA":
            yearly_exposure = yearly_exposure.assign_coords(
                lon=((yearly_exposure.lon + 180) % 360) - 180
            ).sortby("lon")

        # filter to land
        land_gdf = load_land_polygons_for_storm_normalized(
            storm_da=yearly_exposure,
            shapefile_root=SHP_ROOT_NORMALIZED,
            buffer=5.0,
        )
        yearly_exposure = clip_raster_to_land(yearly_exposure, land_gdf)

        # skip if empty
        if yearly_exposure.isnull().all():
            continue

        exposure_list.append(yearly_exposure)
        valid_exposure_ids.append(storm_da.attrs.get("storm_id"))

        # clean up memory
        del exposure, yearly_exposure, land_gdf
        gc.collect()

    return exposure_list, valid_exposure_ids

######################################
#    Per Storm Intensity Functions   #
######################################

def generate_intensity_per_storm(
    haz: TropCyclone,
    centroids: Centroids,
    tc_tracks: TCTracks,
    basin: str,
    wind_threshold: float = 17.0,
) -> tuple[list[xr.DataArray], list[str]]:
    """
    Generate per-storm, per-pixel intensity using CLIMADA haz.intensity.

    Intensity = max wind speed at each pixel over storm lifetime.

    Returns
    -------
    tuple[list[xr.DataArray], list[str]]
        One DataArray per storm with dims ('lat', 'lon') and a list of valid storm IDs
    """

    # --- Grid definition (canonical, no flipping) ---
    lon_vals = np.sort(np.unique(centroids.coord[:, 1]))
    lat_vals = np.sort(np.unique(centroids.coord[:, 0]))

    n_lat = len(lat_vals)
    n_lon = len(lon_vals)

    # validate grid
    assert centroids.coord.shape[0] == n_lat * n_lon

    storm_list_intensity = []
    valid_storm_ids = []

    # load once (avoid repeated toarray calls)
    intensity_all = haz.intensity.toarray()  # shape: (n_storms, n_centroids)

    for i, event in enumerate(tc_tracks.data):


        storm_name = event.name
        storm_id = event.sid
        storm_basin = getattr(event, "basin", None)
        storm_category = getattr(event, "category", None)

        # --- Robust dates ---
        times = event.time
        start_date = pd.to_datetime(times.values[0])
        end_date = pd.to_datetime(times.values[-1])

        try:
            intensity_flat = intensity_all[i, :]
        except Exception as e:
            print(f"⚠️ Could not read intensity for {storm_name}: {e}")
            continue

        if intensity_flat.size != n_lat * n_lon:
            print(
                f"⚠️ Skipping {storm_name}: grid mismatch "
                f"{intensity_flat.size} vs {n_lat * n_lon}"
            )
            continue

        # --- FAST PRE-FILTER (skip weak storms early) ---
        if np.nanmax(intensity_flat) < wind_threshold:
            continue

        # --- Reshape ---
        intensity_2d = intensity_flat.reshape(n_lat, n_lon)

        da = xr.DataArray(
            intensity_2d,
            coords={"lat": lat_vals, "lon": lon_vals},
            dims=["lat", "lon"],
            name=storm_name,
        )

        # Normalize NA basin
        if basin == "NA":
            da = da.assign_coords(
                lon=((da.lon + 180) % 360) - 180
            ).sortby("lon")


        # --- load land gdf for storm ---
        if basin == "NA":
            land_gdf = load_land_polygons_for_storm_normalized(
                storm_da=da,
                shapefile_root=SHP_ROOT_NORMALIZED,
                buffer=5.0,
            )
        else:
            land_gdf = load_land_polygons_for_storm(
                storm_da=da,
                shapefile_gdf_path=GDF_PATH,
                buffer=5.0,
            )
        # --- LANDFALL CHECK ---
        makes_landfall = check_storm_landfall(
            storm_da=da,
            land_gdf=land_gdf,
            wind_threshold=wind_threshold,
        )

        if not makes_landfall:
            continue

        # --- Metadata (aligned with new single-storm version) ---
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
                "during the storm lifetime"
            ),
        })

        # filter to land
        da = clip_raster_to_land(da, land_gdf)

        # if empty no landfall, skip
        if da.isnull().all():
            continue

        storm_list_intensity.append(da)
        valid_storm_ids.append(storm_id)

        # --- Free memory ---
        del da, land_gdf, intensity_flat, intensity_2d
        gc.collect()

    return storm_list_intensity, valid_storm_ids




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

def filter_intensity_by_exposure_ids(
    storm_list_intensity: list[xr.DataArray],
    exposure_valid_ids: list[str],
) -> tuple[list[xr.DataArray], list[str]]:
    """
    Filter intensity storms to only those with valid exposure.

    Returns filtered intensity list and corresponding storm IDs.
    """
    filtered_intensity = []
    filtered_ids = []

    for storm_da in storm_list_intensity:
        storm_id = storm_da.attrs.get("storm_id")
        if storm_id in exposure_valid_ids:
            filtered_intensity.append(storm_da)
            filtered_ids.append(storm_id)

    return filtered_intensity, filtered_ids

#######################################
#    Save Per Storm Functions         #
#######################################
def save_storm_list_intensity(
    storm_list: list[xr.DataArray],
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    save_root: Path = SAVE_ROOT,
):
    """
    Save a list of storm intensity DataArrays as individual NetCDF files in a single directory.
    Each storm is saved using its storm_id as the filename.
    """
    out_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / "intensity"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    chmod_recursive(out_dir, 0o775)

    for da in storm_list:
        da = da.copy()
        da.name = "intensity"
        if da.dtype != "float32":
            da = da.astype("float32")

        storm_id = da.attrs["storm_id"]
        out_file = out_dir / f"{storm_id}.nc"
        da.to_netcdf(out_file, mode="w")
        os.chmod(out_file, 0o775)


def save_storm_list_exposure(
    storm_list: list[xr.DataArray],
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    save_root: Path = SAVE_ROOT,
):
    """
    Save a list of storm exposure DataArrays as individual NetCDF files in a single directory.
    Each storm is saved using its storm_id as the filename.
    """
    out_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / "exposure_hours"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    chmod_recursive(out_dir, 0o775)

    for da in storm_list:
        da = da.copy()
        da.name = "exposure_hours"
        if da.dtype != "float32":
            da = da.astype("float32")

        storm_id = da.attrs["storm_id"]
        out_file = out_dir / f"{storm_id}.nc"
        da.to_netcdf(out_file, mode="w")
        os.chmod(out_file, 0o775)
############################################
#              Main                        #
############################################

def main(
    year: int | str,
    basin: str,
    save_root: Path = SAVE_ROOT,
):
    tracks = TCTracks.from_ibtracs_netcdf(provider="official", basin=basin, year_range=(year, year))
    year = str(year)

    if not tracks.data:
        print(f"No storms found for {basin} in {year}")
        return


    # get centroids for basin
    centroids = generate_basin_centroids(basin, res=RESOLUTION)

    # generate hazard object
    haz = generate_hazard_per_track(tracks, centroids)

    # generate per-storm intensity
    storm_list_intensity, valid_storm_ids, = generate_intensity_per_storm(
        haz=haz,
        centroids=centroids,
        tc_tracks=tracks,
        basin=basin,
        wind_threshold=17.0,
    )
    
    # per storm wind speed
    storm_list_speed = generate_speed_per_storm(
        haz=haz,
        centroids=centroids,
        tc_tracks=tracks,
        valid_storm_ids=valid_storm_ids,
    )

    # generate exposure
    storm_list_exposure, valid_exposure_ids = compute_yearly_exposure_per_storm_list(
        storm_list=storm_list_speed,
        basin=basin,
        wind_threshold=17.0,
    )

    # filter intensity to storms with valid exposure
    storm_list_intensity, filtered_ids = filter_intensity_by_exposure_ids(
        storm_list_intensity=storm_list_intensity,
        exposure_valid_ids=valid_exposure_ids,
    )

    # filter exposure to filterd_ids (precaution)
    storm_list_exposure, _ = filter_intensity_by_exposure_ids(
        storm_list_intensity=storm_list_exposure,
        exposure_valid_ids=filtered_ids,
    )

    # sort both lists by storm_id
    storm_list_intensity = sorted(storm_list_intensity, key=lambda da: da.attrs["storm_id"])
    storm_list_exposure  = sorted(storm_list_exposure, key=lambda da: da.attrs["storm_id"])

    # metadata for saving
    source_id = "ibtracs"
    variant_label = "official"
    experiment_id = "historical"
    batch_year = f"{year}-{year}"

    # save per storm intensity
    save_storm_list_intensity(
        storm_list=storm_list_intensity,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        save_root=save_root,
    )

    # save per storm exposure
    save_storm_list_exposure(
        storm_list=storm_list_exposure,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        save_root=save_root,
    )

    print(f"Saved {len(storm_list_intensity)} intensity and {len(storm_list_exposure)} exposure files for {basin} in {year}.")

main(
    year=year,
    basin=basin,
    save_root=SAVE_ROOT,
)