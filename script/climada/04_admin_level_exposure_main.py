from pathlib import Path
from tempfile import template
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
import rasterra as rt # type: ignore
import pandas as pd  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
import geopandas as gpd  # type: ignore
from affine import Affine  # type: ignore
import os
import warnings
from shapely.geometry import shape  # type: ignore
from collections.abc import Iterator
import argparse
from rasterio.features import shapes  # type: ignore
import dask.array as da  # type: ignore
import warnings
from rra_tools.parallel import run_parallel  # type: ignore
import gc
from typing import NamedTuple
from rasterra import RasterArray  # type: ignore
from shapely.geometry import box
import shapely
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString
from shapely.ops import split, unary_union
import pyarrow.parquet as pq  # type: ignore
import re

warnings.simplefilter("ignore", FutureWarning)


parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--draw_batch", type=str, required=True, help="Draw Batch")


# Parse arguments
args = parser.parse_args()
source_id = args.source_id
variant_label = args.variant_label
experiment_id = args.experiment_id
batch_year = args.batch_year
basin = args.basin
draw_batch = args.draw_batch




# Constants
ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1")
# SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4")
SAVE_ROOT = Path("/mnt/share/scratch/users/mfiking/outputs/stage4a") # TEST

GDF_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0.parquet")
POP_TOTALS_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2021_all_years.parquet")
ANTIMERIDIAN = LineString([(180, -90), (180, 90)])



class StormMeta(NamedTuple):  # type: ignore
    storm_path: Path
    start_year: int
    end_year: int
    storm_id: str


##############################
#     Helper Functions       #
##############################
def normalize_geom_to_0_360(geom):
    """Shift WGS84 geometry from -180–180 to 0–360 for clipping against 0–360 raster."""
    def shift_x(x, y, z=None):
        return x + 360 if x < 0 else x, y
    return shapely.ops.transform(shift_x, geom)

def normalize_longitudes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Shift geometries with x > 180 to x - 360 to avoid wraparound issues."""
    def shift_geom(geom):
        if geom.is_empty:
            return geom
        def shift_coords(x, y, z=None):
            x_new = x - 360 if x > 180 else x
            return (x_new, y) if z is None else (x_new, y, z)
        return shapely.ops.transform(shift_coords, geom)
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(shift_geom)
    return gdf

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


    # Native rasterra clip
    return rr_raster.clip(gdf)

def reproject_raster_to_equal_area(raw_paf_raster):
    raw_paf_raster = raw_paf_raster.to_crs("ESRI:54034")
    # raw_paf_raster = subset_affected_area(raw_paf_raster)

    return raw_paf_raster

def reproject_shapefile_to_equal_area(raw_paf_raster, intersected_shapes, basin):
    maxx = intersected_shapes.geometry.bounds.maxx.max()
    if maxx > 180:
        # geometry crosses the antimeridian
        intersected_shapes = normalize_longitudes(intersected_shapes)

    intersected_shapes = intersected_shapes.to_crs(raw_paf_raster.crs)
    xmin, xmax, ymin, ymax = raw_paf_raster.bounds
    raster_bbox = box(xmin, ymin, xmax, ymax)
    bbox_gdf = gpd.GeoDataFrame(geometry=[raster_bbox], crs=raw_paf_raster.crs)
    intersected_shapes = intersected_shapes.clip(bbox_gdf)
    intersected_shapes["geometry"] = intersected_shapes.geometry.apply(_polygonize)
    
    return intersected_shapes

def clean_raster(raw_raster):

    # Ensure float32 and operate directly on raster array
    raw_raster._ndarray = raw_raster._ndarray.astype(np.float32, copy=False)

    # Convert 0 → NaN in-place
    raw_raster._ndarray[raw_raster._ndarray == 0] = np.nan

    raw_raster = subset_affected_area(raw_raster)

    return raw_raster


def generate_basin_template_raster(basin, res=0.1, buffer_deg=5.0):
    basin_bounds = {
        'EP': ['180E', '0N', '290E', '60N'],
        'NA': ['260E', '0N', '360E', '60N'],
        'NI': ['30E',  '0N', '100E', '50N'],
        'SI': ['20E',  '45S', '100E', '0S'],
        'AU': ['100E', '45S', '180E', '0S'],
        'SP': ['180E', '45S', '250E', '0S'],
        'WP': ['100E', '0N', '180E', '60N'],
    }

    def parse_coord(c):
        match = re.match(r"([0-9\.]+)([ENWS])", c)
        val, hemi = match.groups()
        val = float(val)
        if hemi == 'S': val = -val
        if hemi == 'W': val = 360 - val
        return val

    lon_min, lat_min, lon_max, lat_max = [parse_coord(c) for c in basin_bounds[basin]]

    # Apply buffer
    lon_min -= buffer_deg
    lon_max += buffer_deg
    lat_min -= buffer_deg
    lat_max += buffer_deg

    # Number of rows/cols
    n_cols = int(np.ceil((lon_max - lon_min) / res))
    n_rows = int(np.ceil((lat_max - lat_min) / res))

    # Create empty data array
    data = np.zeros((n_rows, n_cols), dtype=np.float32)

    # Create affine transform: from array index (col,row) to geographic coords
    # Affine: (scale_x, 0, x_min, 0, scale_y, y_max)
    # scale_y is negative because row index increases downward
    transform = Affine(res, 0, lon_min, 0, -res, lat_max)

    # Wrap as RasterArray
    # Wrap as RasterArray
    raster = RasterArray(data=data,
                         transform=transform,
                         crs="EPSG:4326",
                         no_data_value=np.nan
                         )
    return raster

###########################################
#            Storm Functions              #
###########################################

def iter_storms_metadata(draw_store: Path) -> list[StormMeta]:
    """Return a list of StormMeta for each storm in the draw, without loading full data."""
    if not draw_store.exists():
        raise FileNotFoundError(f"Draw store not found: {draw_store}")

    storm_paths = sorted(
        p for p in draw_store.iterdir() if p.is_dir() and p.name.startswith("storm_")
    )

    storms_meta = []
    for storm_path in storm_paths:
        ds = xr.open_zarr(storm_path, consolidated=False, chunks={})  # lazy read, no load
        start_year = pd.to_datetime(ds.attrs["start_date"]).year
        end_year = pd.to_datetime(ds.attrs["end_date"]).year
        storm_id = ds.attrs.get("storm_id", storm_path.name)
        storms_meta.append(StormMeta(storm_path, start_year, end_year, storm_id))
        ds.close()  # close immediately to avoid keeping file handles open
    return storms_meta

def map_storms_to_years(storms_meta: list[StormMeta], years: list[int]):
    storms_by_year = {year: [] for year in years}
    for storm in storms_meta:
        for year in range(storm.start_year, storm.end_year + 1):
            if year in storms_by_year:
                storms_by_year[year].append(storm.storm_path)
    return storms_by_year

##############################
#     Load Raw PAF Raster    #
##############################

def chmod_recursive(path: Path, mode: int = 0o775):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)

def get_draw_zarr_path(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    metric: str,
) -> Path:
    """
    Locate draw-level storm Zarr store produced by Stage 1.
    """
    start_year, end_year = batch_year.split("-")
    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    metrics_allowed = ["intensity", "exposure_hours", "days_impact"]
    if metric not in metrics_allowed:
        raise ValueError(f"Invalid metric: {metric}. Allowed: {metrics_allowed}")
    
    draw_store = (
        ROOT
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / metric
        / f"{metric}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
    )

    if not draw_store.exists():
        raise FileNotFoundError(f"Zarr store not found: {draw_store}")

    return draw_store


def get_exposure_storm_from_draw(
    draw_store: Path,
    storm_id: str,
) -> xr.Dataset:
    """
    Retrieve a specific storm Dataset from a draw Zarr store.

    Parameters
    ----------
    draw_store : Path
        Path to the draw-level Zarr store.
    storm_id : str
        Identifier of the storm to retrieve.

    Returns
    -------
    xr.Dataset
        Dataset for the specified storm.
    """

    # format storm_id to match 4 digit format (e.g., 1 -> 0001)
    storm_id = f"{int(storm_id):04d}"

    storm_path = draw_store / f"storm_{storm_id}"
    if not storm_path.exists():
        raise FileNotFoundError(f"Storm {storm_id} not found in draw store {draw_store}")
    
    return xr.open_zarr(
        storm_path,
        consolidated=False,
        chunks="auto",   # critical for raster ops
        decode_times=False,
    )


##########################################
#           Load Shapefile               #
##########################################

def load_shapefiles():
    shapefile=gpd.read_parquet(GDF_PATH)

    return shapefile


##########################################
#           Load Population              #
##########################################
def load_population_dataframe():
    pop_df = pd.read_parquet(POP_TOTALS_PATH)

    return pop_df

def get_population_total(pop_df: pd.DataFrame, year: int, admin_id: int):

    subset = pop_df[
        (pop_df["year_id"] == year) &
        (pop_df["location_id"] == admin_id) &
        (pop_df["age_group_id"] == 22) &
        (pop_df["sex_id"] == 3)
    ]

    if subset.empty:
        return 1.0, True  # sentinel population but explicitly flagged

    if len(subset) > 1:
        raise ValueError(
            f"Multiple population rows found for admin_id={admin_id}, year={year}"
        )

    return subset["population"].item(), False


#######################################
#      Read in Gridded Population     #
#######################################


def load_in_gridded_population(year: int | str, meters: int | str, bounds: tuple | None = None):
    pop_path = Path("/mnt/team/rapidresponse/pub/population-model/results/current/")
    
    if bounds is None:
        pop_raster = rt.load_raster(pop_path / f"world_cylindrical_{meters}" / f"{year}q1.tif")
        return pop_raster
    else:
        pop_raster = rt.load_raster(pop_path / f"world_cylindrical_{meters}" / f"{year}q1.tif", 
                                bounds=bounds)
    
    return pop_raster

##########################################
#     Intersect Shapefiles with data     #
##########################################
def _polygonize(geom):
    if geom is None:
        return None

    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom

    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        return unary_union(polys)

    return None


def intersect_shapefile_with_raster(
    shapefile_gdf: gpd.GeoDataFrame,
    rr_raster,  # RasterArray
    buffer_degrees: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Find shapefile features that intersect areas with RR > 0 in a raster.

    Parameters
    ----------
    shapefile_gdf : geopandas.GeoDataFrame
        Input polygons
    rr_raster : RasterArray
        Relative risk raster (GeoTIFF-derived)
    buffer_degrees : float, optional
        Optional buffer applied to RR mask geometry (degrees)

    Returns
    -------
    geopandas.GeoDataFrame
        Subset of shapefile intersecting RR>0 areas
    """


    # compute raster bounds
    height, width = rr_raster._ndarray.shape
    transform = rr_raster.transform
    xmin, ymin = transform * (0, height)
    xmax, ymax = transform * (width, 0)
    raster_bbox = box(xmin, ymin, xmax, ymax)
    shapefile_gdf = shapefile_gdf.clip(raster_bbox)

    shapefile_gdf["geometry"] = shapefile_gdf.geometry.apply(_polygonize)
    shapefile_gdf = shapefile_gdf.dropna(subset=["geometry"]).reset_index(drop=True)
    # ------------------------------
    # 1. Build RR>0 mask
    # ------------------------------
    rr_data = rr_raster._ndarray
    mask = rr_data > 0 # we already converted 0s to nans, so > 0 excludes all nans

    if not mask.any():
        print("⚠️ No nonzero RR pixels found")
        return shapefile_gdf.iloc[0:0].copy()

    # ------------------------------
    # 2. Convert mask → polygons
    # ------------------------------
    shapes_gen = shapes(
        mask.astype(np.uint8),
        transform=rr_raster.transform,
    )

    geometries = [
        shape(geom).buffer(0) for geom, value in shapes_gen if value == 1
    ]

    rr_geom = unary_union(geometries)

    if isinstance(rr_geom, GeometryCollection):
        polys = [g for g in rr_geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        rr_geom = unary_union(polys)

    if buffer_degrees > 0:
        rr_geom = rr_geom.buffer(buffer_degrees)

    rr_gdf = gpd.GeoDataFrame(geometry=[rr_geom], crs=rr_raster.crs)

    # ------------------------------
    # 3. CRS alignment
    # ------------------------------
    if shapefile_gdf.crs != rr_gdf.crs:
        shapefile_gdf = shapefile_gdf.to_crs(rr_gdf.crs)

    # ------------------------------
    # 4. Spatial intersection
    # ------------------------------
    intersected = shapefile_gdf[
        shapefile_gdf.intersects(rr_geom)
    ].copy().reset_index(drop=True)

    print(
        f"Found {len(intersected)} shapefile features intersecting RR raster"
    )

    return intersected



###########################################
#             Save Functions              #
###########################################
def save_draw_dataframe(
    exposure_df: pd.DataFrame,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    save_root: Path = SAVE_ROOT,
):
    """
    Save batch-year admin level exposure dataframe to Parquet.

    Expected paf_df columns (minimum):
        - storm_id
        - year
        - location_id
        - person_storm_hours
        - total_population
        - max_wind_speed
    """

    save_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / "admin_level_exposure"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    start_year, end_year = batch_year.split("-")

    filename = f"admin_level_exposure_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.parquet"
    save_path = save_dir / filename

    # Optional: enforce consistent column order
    preferred_cols = [
        "draw",
        "location_id",
        "year",
        "storm_id",
        "person_storm_hours",
        "total_population",
        "max_wind_speed",
    ]
    existing_cols = [c for c in preferred_cols if c in exposure_df.columns]
    other_cols = [c for c in exposure_df.columns if c not in existing_cols]
    exposure_df = exposure_df[existing_cols + other_cols]

    exposure_df.to_parquet(save_path, index=False)

    print(f"Saved batch-year exposure dataframe: {save_path}")

##########################################
#       Antemeridian Function            #
##########################################
def split_antimeridian_geom(geom_cea):

    gdf = gpd.GeoSeries([geom_cea], crs="ESRI:54034")
    geom_ll = gdf.to_crs("EPSG:4326").iloc[0]

    minx, _, maxx, _ = geom_ll.bounds

    # no crossing
    if maxx - minx <= 180:
        return [geom_cea]

    result = split(geom_ll, ANTIMERIDIAN)

    pieces = [g for g in result.geoms if g.geom_type in ("Polygon", "MultiPolygon")]

    west = []
    east = []

    for g in pieces:
        if g.centroid.x < 0:
            west.append(g)
        else:
            east.append(g)

    west_union = unary_union(west) if west else None
    east_union = unary_union(east) if east else None

    out_ll = [g for g in [west_union, east_union] if g is not None]

    out_cea = (
        gpd.GeoSeries(out_ll, crs="EPSG:4326")
        .to_crs("ESRI:54034")
        .tolist()
    )

    return out_cea


#############################
# Check if draw is complete #
#############################
def check_if_draw_is_complete(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    save_root: Path = SAVE_ROOT,
) -> bool:
    """
    Return True if the admin-level exposure Parquet for a given draw exists and is valid.

    Returns False if missing, zero-byte, unreadable, or empty.
    """
    save_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / "admin_level_exposure"
    )

    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    start_year, end_year = batch_year.split("-")

    filename = (
        f"admin_level_exposure_{basin}_{source_id}_{experiment_id}_"
        f"{variant_label}_{start_year}01_{end_year}12{draw_text}.parquet"
    )

    save_path = save_dir / filename

    # 1. File must exist
    if not save_path.exists():
        return False

    # 2. File must not be zero bytes
    if save_path.stat().st_size == 0:
        return False

    # 3. File must be readable
    try:
        pf = pq.ParquetFile(save_path)
    except Exception:
        return False

    # 4. File must have rows
    if pf.metadata.num_rows == 0:
        return False

    return True

###########################################
#                Main                     #
###########################################
def process_single_draw(args):
    (
        source_id,
        variant_label,
        experiment_id,
        batch_year,
        basin,
        draw,
        template_raster,
    ) = args

    # ---- skip if already complete ----
    if check_if_draw_is_complete(
        source_id,
        variant_label,
        experiment_id,
        batch_year,
        basin,
        draw,
    ):
        print(f"Skipping draw: {draw} (already complete)")
        return None

    intensity_draw_store = get_draw_zarr_path(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        metric="intensity",
    )

    exposure_draw_store = get_draw_zarr_path(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        metric="exposure_hours",
    )

    shapefile = load_shapefiles()
    pop_df = load_population_dataframe()

    # Parse year range from batch_year (same as Stage 2)
    start_year, end_year = map(int, batch_year.split("-"))
    all_years = list(range(start_year, end_year + 1))

    # ---- Handle empty intensity draws ----
    if intensity_draw_store is None or not any(intensity_draw_store.iterdir()):
        print(f"⚠️ Draw {draw} has no intensity data.")
        return None
    
    exposure_df_list = []

    # ---------------------------------------------------
    # STEP 1: Load storm metadata only
    # ---------------------------------------------------
    storms_meta = iter_storms_metadata(intensity_draw_store)
    storms_by_year = map_storms_to_years(storms_meta, all_years)

    # ---------------------------------------------------
    # STEP 2: Process year-by-year
    # ---------------------------------------------------
    for year in all_years[:1]:
        print(f"Processing year: {year}")

        storm_paths_in_year = storms_by_year[year]
        print(f"Number of storms: {len(storm_paths_in_year)}")

        if not storm_paths_in_year:
            continue

        for storm_path in storm_paths_in_year:

            storm_ds = xr.open_zarr(storm_path, consolidated=False, chunks="auto")
            storm_id = storm_ds.attrs.get("storm_id", storm_path.name)
            print(f"Processing strom id: {storm_id}")

            # ---------------------------------------------------
            # Load exposure
            # ---------------------------------------------------
            try:
                storm_exposure = get_exposure_storm_from_draw(
                    exposure_draw_store,
                    storm_id,
                )
                print(f"Exposure loaded")
            except (FileNotFoundError, KeyError):
                storm_ds.close()
                continue


            print(f"starting raster creation")
            # ---------------------------------------------------
            # Rasterize storm fields
            # ---------------------------------------------------
            intensity_raster = to_raster(
                ds=storm_ds["intensity"].astype("float32"),  # keep lazy
                no_data_value=np.nan,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326"
            ).resample_to(target=template_raster, resampling="nearest")

            print(f"intensity raster created")
            # ---------------------------------------------------
            # Rasterize exposure directly from the lazy DataArray
            # ---------------------------------------------------
            exposure_raster = to_raster(
                ds=storm_exposure["exposure_hours"].astype("float32"),  # keep lazy
                no_data_value=np.nan,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326"
            ).resample_to(target=template_raster, resampling="nearest")
            print("Intensity and exposure rasterized")            
            
            # Clean up
            storm_ds.close()
            del storm_exposure
            del storm_ds


            intensity_raster._ndarray = intensity_raster._ndarray
            exposure_raster._ndarray = exposure_raster._ndarray
            print(f"Computed raster data")

            intensity_raster = clean_raster(intensity_raster)
            exposure_raster = clean_raster(exposure_raster)
            print(f"Rasters Cleaned")

            # ---------------------------------------------------
            # Subset admin shapes intersecting storm
            # ---------------------------------------------------
            intersected_shapes = intersect_shapefile_with_raster(
                shapefile,
                exposure_raster,
                buffer_degrees=0.0,
            )
            if intersected_shapes.empty:
                print("No intersecting shapes for storm")
                del intensity_raster, exposure_raster
                del intersected_shapes
                gc.collect()
                continue
            # ---------------------------------------------------
            # Generate max windspeed before reprojection
            # ---------------------------------------------------
            max_wind_by_loc = {}

            for _, admin_shape in intersected_shapes.iterrows():
                admin_id = admin_shape["loc_id"]
                admin_geom = admin_shape.geometry

                # Clip intensity in WGS84
                admin_intensity = intensity_raster.clip(admin_geom).mask(admin_geom, all_touched=True)
                arr = admin_intensity._ndarray

                if np.isfinite(arr).any():
                    max_wind_by_loc[admin_id] = float(np.nanmax(arr))
                else:
                    max_wind_by_loc[admin_id] = np.nan

                del admin_intensity, arr
                
            print(f"Max windspeed calculated")

            intensity_raster = reproject_raster_to_equal_area(intensity_raster)
            intersected_shapes = reproject_shapefile_to_equal_area(intensity_raster, intersected_shapes, basin)

            print("Shapefile reprojected")
            # ---------------------------------------------------
            # STEP 3: Admin-level calculations
            # ---------------------------------------------------
            for _, admin_shape in intersected_shapes.iterrows():

                admin_geom = admin_shape.geometry
                admin_id = admin_shape["loc_id"]
                print(f"Processing admin id: {admin_id}")

                geom_pieces = split_antimeridian_geom(admin_geom)
                print(f"Number of pieces: {len(geom_pieces)}")

                if not geom_pieces:
                    geom_pieces = [admin_geom]

                person_storm_hours_total = 0.0
                population_exposed_total = 0.0

                pop_sum, special_region_flag = get_population_total(
                    pop_df=pop_df,
                    year=year,
                    admin_id=admin_id,
                )

                for piece_geom in geom_pieces:
                    bounds = piece_geom.bounds
                    piece_geom_wgs84 = gpd.GeoSeries([piece_geom], crs="ESRI:54034").to_crs(exposure_raster.crs).iloc[0]
                    
                    piece_geom_wgs84 = normalize_geom_to_0_360(piece_geom_wgs84)
                    admin_exposure = exposure_raster.clip(piece_geom_wgs84).mask(piece_geom_wgs84, all_touched=True)

                    # ----------------------------
                    # Load population
                    # ----------------------------
                    pop_raster = load_in_gridded_population(
                        year,
                        100,
                        bounds=bounds,
                    )

                    pop_raster = pop_raster.mask(piece_geom, all_touched=True)
                    pop_arr = pop_raster._ndarray

                    exposure_resampled = admin_exposure.resample_to(pop_raster)

                    exposure_resampled = exposure_resampled.mask(piece_geom, all_touched=True)

                    exposure_arr = exposure_resampled._ndarray

                    # ----------------------------
                    # Valid cells
                    # ----------------------------
                    pop_mask = (pop_arr > 0) & np.isfinite(pop_arr)
                    exposure_mask = (exposure_arr > 0) & np.isfinite(exposure_arr)
                    valid_mask = pop_mask & exposure_mask

                    if valid_mask.any():
                        person_storm_hours_total += (
                            pop_arr[valid_mask] * exposure_arr[valid_mask]
                        ).sum()

                        population_exposed_total += pop_arr[valid_mask].sum()
                        print(f"Pop exposed: {population_exposed_total}")
                        
                    # cleanup piece memory
                    del pop_raster
                    del exposure_resampled
                    del pop_arr, exposure_arr
                    del pop_mask, exposure_mask, valid_mask
                    
                max_wind_speed = max_wind_by_loc[admin_id]
                exposure_df_list.append(
                    {
                        "draw": draw,
                        "storm_id": storm_id,
                        "year": year,
                        "location_id": admin_id,
                        "person_storm_hours": float(person_storm_hours_total),
                        "total_population": float(pop_sum),
                        "total_population_exposed": float(population_exposed_total),
                        "max_wind_speed": float(max_wind_speed),
                        "special_region_flag": special_region_flag,
                    }
                )

            # ---------------------------------------------------
            # storm cleanup
            # ---------------------------------------------------

            del intensity_raster
            del exposure_raster
            del intersected_shapes

            gc.collect()

    if not exposure_df_list:
        return None

    draw_df = pd.DataFrame.from_records(exposure_df_list)

    save_draw_dataframe(
        exposure_df=draw_df,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    )

    draw_df = None
    exposure_df_list = []

    return draw_df



def main(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw_batch: str,
):
    start_draw, end_draw = map(int, draw_batch.split("-"))
    draws = list(range(start_draw, end_draw + 1))

    
    # Generate basin-wide template raster once
    template_raster = generate_basin_template_raster(basin, res=0.1)

    draw_args = [
        (
            source_id,
            variant_label,
            experiment_id,
            batch_year,
            basin,
            draw,
            template_raster,
        )
        for draw in draws
    ]

    results = run_parallel(
        runner=process_single_draw,
        arg_list=draw_args,
        num_cores=1,
    )
    print("Completed Parallel Tasks")

    draw_dir = (
        SAVE_ROOT
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / "admin_level_exposure"
    )

    chmod_recursive(draw_dir, mode=0o775)
    
main(
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
    batch_year=batch_year,
    basin=basin,
    draw_batch=draw_batch,
)
    