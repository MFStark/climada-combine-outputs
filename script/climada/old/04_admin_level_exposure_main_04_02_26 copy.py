from pathlib import Path
from tempfile import template
from tracemalloc import start
from matplotlib.image import resample
from matplotlib.pylab import f
from netCDF4 import num2date
import xarray as xr  # type: ignore
import numpy as np
from zarr import save  # type: ignore
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
import rasterio  # type: ignore

warnings.simplefilter("ignore", FutureWarning)


parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--draw_batch", type=str, required=True, help="Draw Batch")
parser.add_argument("--num_cores", type=int, default=5, help="Number of cores for parallel processing")


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
ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1_v2")
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4_v2")
# SAVE_ROOT = Path("/mnt/share/scratch/users/mfiking/outputs/stage4_v2") # TEST

GDF_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0.parquet")
POP_TOTALS_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2021_all_years.parquet")
ANTIMERIDIAN = LineString([(180, -90), (180, 90)])
SHP_ROOT_NORMALIZED = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29')



class StormMeta(NamedTuple):  # type: ignore
    storm_path: Path
    start_year: int
    end_year: int
    storm_id: str


##############################
#     Helper Functions       #
##############################
def ensure_min_grid(da: xr.DataArray, buffer_deg: float = 0.1) -> xr.DataArray:
    if da.lat.size == 1:
        c = float(da.lat.values[0])
        da = da.reindex(lat=[c-buffer_deg, c, c+buffer_deg], fill_value=0)

    if da.lon.size == 1:
        c = float(da.lon.values[0])
        da = da.reindex(lon=[c-buffer_deg, c, c+buffer_deg], fill_value=0)

    return da

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

def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.assign_coords(
        lon=(((ds.lon + 180) % 360) - 180)
    ).sortby("lon")

    return ds

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

def reproject_shapefile_to_equal_area(
    rr_raster,
    exposure_raster,
    intersected_shapes,
    basin,
):
    """
    Reproject intersected shapes to equal-area CRS (ESRI:54034),
    using the combined extent of RR and exposure rasters for safe clipping.
    """

    # --- Normalize longitudes if crossing antimeridian ---
    maxx = intersected_shapes.geometry.bounds.maxx.max()
    if maxx > 180:
        intersected_shapes = normalize_longitudes(intersected_shapes)

    # --- Attempt to reproject raster(s) first ---
    raster_reprojected = False
    try:
        rr_cea = reproject_raster_to_equal_area(rr_raster)
        exposure_cea = reproject_raster_to_equal_area(exposure_raster)
        raster_reprojected = True
    except Exception as e:
        print(f"[WARNING] Raster reprojection failed ({e}) → will skip raster-based clipping")

    # --- Compute combined raster bounds if reprojection succeeded ---
    if raster_reprojected:
        intersected_shapes = intersected_shapes.to_crs(rr_cea.crs)

        try:
            def _bounds(r):
                h, w = r._ndarray.shape
                t = r.transform
                xmin, ymin = t * (0, h)
                xmax, ymax = t * (w, 0)
                return xmin, ymin, xmax, ymax

            xmin1, ymin1, xmax1, ymax1 = _bounds(rr_cea)
            xmin2, ymin2, xmax2, ymax2 = _bounds(exposure_cea)

            combined_bbox = box(
                min(xmin1, xmin2),
                min(ymin1, ymin2),
                max(xmax1, xmax2),
                max(ymax1, ymax2),
            )

            bbox_gdf = gpd.GeoDataFrame(geometry=[combined_bbox], crs=rr_cea.crs)
            clipped = intersected_shapes.clip(bbox_gdf)

            if clipped.empty:
                print("[INFO] Clipping would empty intersected_shapes → skipping clip")
            else:
                intersected_shapes = clipped
                intersected_shapes["geometry"] = intersected_shapes.geometry.apply(_polygonize)
        except Exception as e:
            print(f"[WARNING] Clipping failed ({e}) → skipping clip")
    else:
        # Raster failed, just reproject shapes directly to equal area CRS
        intersected_shapes = intersected_shapes.to_crs("ESRI:54034")
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

    # --- Normalize NA basin to -180 → 180 ---
    if basin == "NA":
        lon_min = ((lon_min + 180) % 360) - 180
        lon_max = ((lon_max + 180) % 360) - 180

        # ensure monotonic bounds
        if lon_max <= lon_min:
            lon_max += 360


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


def load_shapefiles_normalized(
    shapefile_root: Path = SHP_ROOT_NORMALIZED,
) -> gpd.GeoDataFrame:
    """
    Load land polygons intersecting storm bounding box (-180 to 180 longitude space).
    """
    admin_level = 0
    simplified_suffix = ''

    # --- Load shapefile ---
    shp_path = shapefile_root / f"lbd_standard_admin_{admin_level}{simplified_suffix}.shp"
    gdf = gpd.read_file(shp_path)

    return gdf
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


def intersect_shapefile_with_rasters(
    shapefile_gdf: gpd.GeoDataFrame,
    rr_raster,                 # intensity / RR raster
    exposure_raster,           # exposure raster
    buffer_degrees: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Return subset of shapefile features intersecting RR > 0 areas,
    using the combined spatial extent of RR and exposure rasters.
    """

    # ------------------------------
    # 0. Validate CRS
    # ------------------------------
    if shapefile_gdf.crs is None:
        shapefile_gdf = shapefile_gdf.set_crs(rr_raster.crs)

    if rr_raster.crs != exposure_raster.crs:
        raise ValueError("RR raster and exposure raster must share the same CRS")

    # ------------------------------
    # 1. Compute combined raster extent
    # ------------------------------
    def _bounds(r):
        h, w = r._ndarray.shape
        t = r.transform
        xmin, ymin = t * (0, h)
        xmax, ymax = t * (w, 0)
        return xmin, ymin, xmax, ymax

    xmin1, ymin1, xmax1, ymax1 = _bounds(rr_raster)
    xmin2, ymin2, xmax2, ymax2 = _bounds(exposure_raster)

    combined_bbox = box(
        min(xmin1, xmin2),
        min(ymin1, ymin2),
        max(xmax1, xmax2),
        max(ymax1, ymax2),
    )

    # ------------------------------
    # 2. Early clip (cheap + critical)
    # ------------------------------
    shapefile_gdf = shapefile_gdf.clip(combined_bbox)

    if shapefile_gdf.empty:
        return shapefile_gdf

    # Clean geometries
    shapefile_gdf["geometry"] = shapefile_gdf.geometry.apply(_polygonize)
    shapefile_gdf = shapefile_gdf.dropna(subset=["geometry"]).reset_index(drop=True)

    # ------------------------------
    # 3. Build RR mask
    # ------------------------------
    rr_data = rr_raster._ndarray
    mask = rr_data > 0

    if not mask.any():
        return shapefile_gdf.iloc[0:0].copy()

    # ------------------------------
    # 4. Mask → geometry
    # ------------------------------
    shapes_gen = shapes(mask.astype(np.uint8), transform=rr_raster.transform)

    geoms = [shape(geom) for geom, val in shapes_gen if val == 1]

    if not geoms:
        return shapefile_gdf.iloc[0:0].copy()

    rr_geom = unary_union(geoms).buffer(0)

    # Normalize geometry type
    if isinstance(rr_geom, GeometryCollection):
        rr_geom = unary_union([
            g for g in rr_geom.geoms if isinstance(g, (Polygon, MultiPolygon))
        ])

    if buffer_degrees > 0:
        rr_geom = rr_geom.buffer(buffer_degrees)

    # ------------------------------
    # 5. CRS alignment
    # ------------------------------
    if shapefile_gdf.crs != rr_raster.crs:
        shapefile_gdf = shapefile_gdf.to_crs(rr_raster.crs)

    # ------------------------------
    # 6. Final intersection
    # ------------------------------
    intersected = shapefile_gdf[
        shapefile_gdf.intersects(rr_geom)
    ].copy().reset_index(drop=True)

    return intersected


##########################################
#       Antemeridian Function            #
##########################################
def split_antimeridian_geom(geom_cea):
    gdf = gpd.GeoSeries([geom_cea], crs="ESRI:54034")
    geom_ll = gdf.to_crs("EPSG:4326").iloc[0]

    minx, _, maxx, _ = geom_ll.bounds

    # no crossing
    if not (maxx > 180 or minx < -180 or (maxx - minx > 180)):
        return [geom_cea], [geom_ll]  # <--- return both CEA and WGS84

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

    # WGS84 output
    out_wgs84 = out_ll

    return out_cea, out_wgs84



###########################################
#             Save Functions              #
###########################################

def save_storm_exposure(
    exposure_df: pd.DataFrame,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    year: int,
    draw: int,
    storm_id: str | int,
    save_root: Path = SAVE_ROOT,
):
    save_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"tc_risk_draw_{draw}"
        / "storm_exposure"
        / str(year)
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    
    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    start_year, end_year = batch_year.split("-")

    file_name = f"storm_{storm_id}_{basin}_{source_id}_{variant_label}_{experiment_id}_{start_year}01_{end_year}12{draw_text}.parquet"

    save_path = save_dir / file_name

    exposure_df.to_parquet(save_path, index=False)

def save_yearly_exposure(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    year: int,
    draw: int,
    save_root: Path = SAVE_ROOT,
):
    save_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"tc_risk_draw_{draw}"
        / "yearly_exposure"
        / str(year)
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    start_year, end_year = batch_year.split("-")

    file_name = f"yearly_exposure_{year}_{basin}_{source_id}_{variant_label}_{experiment_id}_{start_year}01_{end_year}12{draw_text}.parquet"

    save_path = save_dir / file_name

    storm_dir = save_root / source_id / variant_label / experiment_id / batch_year / basin / f"tc_risk_draw_{draw}" / "storm_exposure" / str(year)
    pattern = f"*{draw_text}.parquet" if draw_text else "*.parquet"
    storm_files = list(storm_dir.glob(pattern))

    if not storm_files:
        print(f"⚠️ No storm exposure files found for year {year} in {storm_dir}")
        return None
    
    dfs = []

    for storm_file in storm_files:
        try:
            dfs.append(pd.read_parquet(storm_file))
        except Exception as e:
            print(f"Error reading {storm_file}: {e}")

    if not dfs:
        print(f"⚠️ No valid storm dataframes for year {year}")
        return None

    yearly_df = pd.concat(dfs, ignore_index=True)

    yearly_df.to_parquet(save_path, index=False)



def save_draw_dataframe(
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
        / f"tc_risk_draw_{draw}"
        / "admin_level_exposure"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    start_year, end_year = batch_year.split("-")

    filename = f"admin_level_exposure_{basin}_{source_id}_{variant_label}_{experiment_id}_{start_year}01_{end_year}12{draw_text}.parquet"
    save_path = save_dir / filename

    all_year_dir = save_root / source_id / variant_label / experiment_id / batch_year / basin / f"tc_risk_draw_{draw}" / "yearly_exposure" 

    draw_dfs = []
    for year in range(int(start_year), int(end_year) + 1):
        year_dir = all_year_dir / str(year)
        if not year_dir.exists():
            print(f"⚠️ Yearly exposure directory not found: {year_dir}")
            continue
        pattern = f"*{draw_text}.parquet" if draw_text else "*.parquet"
        year_files = list(year_dir.glob(pattern))
        if not year_files:
            print(f"⚠️ No yearly exposure files found for year {year} in {year_dir}")
            continue
        for year_file in year_files:
            try:
                df = pd.read_parquet(year_file)
                draw_dfs.append(df)
            except Exception as e:
                print(f"Error reading {year_file}: {e}")

    if not draw_dfs:
        print(f"⚠️ No valid yearly dataframes found for draw {draw}")
        return None
    draw_df = pd.concat(draw_dfs, ignore_index=True)
    draw_df.to_parquet(save_path, index=False)


################################
# Check Completion Functions   #
################################

def check_if_storm_is_complete(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    year: int,
    draw: int,
    storm_id: str | int,
    save_root: Path = SAVE_ROOT,
) -> bool:
    """
    Return True if a single storm-level exposure Parquet exists and is valid.
    Performs a lightweight read to catch partial/corrupt files.
    """
    save_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"tc_risk_draw_{draw}"
        / "storm_exposure"
        / str(year)
    )

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    filename = (
        f"storm_{storm_id}_{basin}_{source_id}_{variant_label}_"
        f"{experiment_id}_{start_year}01_{end_year}12{draw_text}.parquet"
    )
    save_path = save_dir / filename

    # ----------------------------
    # Basic existence checks
    # ----------------------------
    if not save_path.exists() or save_path.stat().st_size == 0:
        return False

    # ----------------------------
    # Robust parquet validation
    # ----------------------------
    try:
        pf = pq.ParquetFile(save_path)

        # Check metadata rows
        if pf.metadata is None or pf.metadata.num_rows == 0:
            return False

        # Attempt to read a small portion of data
        first_cols = pf.schema.names[:1]  # minimal column read
        _ = pf.read_row_group(0, columns=first_cols)

    except Exception:
        return False

    return True

def check_if_year_is_complete(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    year: int,
    draw: int,
    save_root: Path = SAVE_ROOT,
) -> bool:
    """
    Return True if a yearly exposure Parquet exists and is valid.
    Performs a minimal read to detect partial or corrupted files.
    """
    save_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"tc_risk_draw_{draw}"
        / "yearly_exposure"
        / str(year)
    )

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    filename = (
        f"yearly_exposure_{year}_{basin}_{source_id}_{variant_label}_"
        f"{experiment_id}_{start_year}01_{end_year}12{draw_text}.parquet"
    )
    save_path = save_dir / filename

    # ----------------------------
    # Basic existence check
    # ----------------------------
    if not save_path.exists() or save_path.stat().st_size == 0:
        return False

    # ----------------------------
    # Robust parquet validation
    # ----------------------------
    try:
        pf = pq.ParquetFile(save_path)

        # Check metadata
        if pf.metadata is None or pf.metadata.num_rows == 0:
            return False

        # Attempt to read a small portion of data
        first_cols = pf.schema.names[:1]  # minimal column read
        _ = pf.read_row_group(0, columns=first_cols)

    except Exception:
        return False

    return True

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

    Returns False if missing, zero-byte, unreadable, or empty/corrupted.
    """
    save_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"tc_risk_draw_{draw}"
        / "admin_level_exposure"
    )

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    filename = (
        f"admin_level_exposure_{basin}_{source_id}_{variant_label}_{experiment_id}_"
        f"{start_year}01_{end_year}12{draw_text}.parquet"
    )
    save_path = save_dir / filename

    # ----------------------------
    # 1. File existence & size
    # ----------------------------
    if not save_path.exists() or save_path.stat().st_size == 0:
        return False

    # ----------------------------
    # 2. Parquet readability & row check
    # ----------------------------
    try:
        pf = pq.ParquetFile(save_path)
        if pf.metadata is None or pf.metadata.num_rows == 0:
            return False

        # Attempt minimal read to detect corruption
        first_cols = pf.schema.names[:1]  # only first column
        _ = pf.read_row_group(0, columns=first_cols)

    except Exception:
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
    print(f"Processing draw: {draw} for batch_year: {batch_year}")
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

    if basin == "NA":
        shapefile = load_shapefiles_normalized()
    else:
        shapefile = load_shapefiles()
    pop_df = load_population_dataframe()

    # Parse year range from batch_year (same as Stage 2)
    start_year, end_year = map(int, batch_year.split("-"))
    all_years = list(range(start_year, end_year + 1))

    # ---- Handle empty intensity draws ----
    if intensity_draw_store is None or not any(intensity_draw_store.iterdir()):
        print(f"⚠️ Draw {draw} has no intensity data.")
        return None
    

    # ---------------------------------------------------
    # STEP 1: Load storm metadata only
    # ---------------------------------------------------
    storms_meta = iter_storms_metadata(intensity_draw_store)
    storms_by_year = map_storms_to_years(storms_meta, all_years)

    # ---------------------------------------------------
    # STEP 2: Process year-by-year
    # ---------------------------------------------------
    for year in all_years:
        print(f"Processing year: {year}")

        # check if yearly exposure already complete
        if check_if_year_is_complete(
            source_id,
            variant_label,
            experiment_id,
            batch_year,
            basin,
            year,
            draw,
        ):
            print(f"Skipping year: {year} (already complete)")
            continue

        storm_paths_in_year = storms_by_year[year]
        print(f"Number of storms: {len(storm_paths_in_year)}")

        if not storm_paths_in_year:
            continue

        for storm_path in storm_paths_in_year:
            storm_records = []

            storm_ds = xr.open_zarr(storm_path, consolidated=False, chunks="auto")
            storm_id = storm_ds.attrs.get("storm_id", storm_path.name)
            print(f"Processing storm id: {storm_id}")

            # check if storm-level exposure already complete
            if check_if_storm_is_complete(
                source_id=source_id,
                variant_label=variant_label,
                experiment_id=experiment_id,
                batch_year=batch_year,
                basin=basin,
                year=year,
                storm_id=storm_id,
                draw=draw,
            ):
                print(f"Skipping storm id: {storm_id} (already complete)")
                storm_ds.close()
                continue
            # ---------------------------------------------------
            # Load exposure
            # ---------------------------------------------------
            try:
                storm_exposure = get_exposure_storm_from_draw(
                    exposure_draw_store,
                    storm_id,
                )
                # print(f"Exposure loaded")
            except (FileNotFoundError, KeyError):
                storm_ds.close()
                continue
            
            # Normalize longitudes if basin is NA
            if basin == "NA":
                storm_ds = normalize_dataset(storm_ds)
                storm_exposure = normalize_dataset(storm_exposure)
                
            # print(f"starting raster creation")

            #----------------------------
            # Ensure min grid
            #----------------------------
            storm_ds = ensure_min_grid(storm_ds.astype("float32"))
            storm_exposure = ensure_min_grid(storm_exposure.astype("float32"))

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

            # print(f"intensity raster created")
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
            # print("Intensity and exposure rasterized")            
            
            # Clean up
            storm_ds.close()
            del storm_exposure
            del storm_ds


            intensity_raster._ndarray = intensity_raster._ndarray
            exposure_raster._ndarray = exposure_raster._ndarray
            # print(f"Computed raster data")

            try: 
                intensity_raster = clean_raster(intensity_raster)
            except ValueError:
                print("No affected pixels found in intensity raster → skipping storm")

            try:
                exposure_raster = clean_raster(exposure_raster)
            except ValueError:
                print("No affected pixels found in exposure raster → skipping storm")
                del intensity_raster, exposure_raster
                gc.collect()
                continue

            # print(f"Rasters Cleaned")

            # ---------------------------------------------------
            # Subset admin shapes intersecting storm
            # ---------------------------------------------------
            intersected_shapes = intersect_shapefile_with_rasters(
                shapefile,
                intensity_raster,
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

                try:
                    admin_intensity = intensity_raster.clip(admin_geom)

                    # Only mask if needed for precision at edges
                    admin_intensity = admin_intensity.mask(admin_geom, all_touched=True)

                    arr = admin_intensity._ndarray

                    max_wind_by_loc[admin_id] = (
                        float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan
                    )

                except rasterio.errors.WindowError:
                    # more precise than ValueError
                    max_wind_by_loc[admin_id] = np.nan

                finally:
                    if "admin_intensity" in locals():
                        del admin_intensity
                    if "arr" in locals():
                        del arr

            intersected_shapes = reproject_shapefile_to_equal_area(intensity_raster, exposure_raster, intersected_shapes, basin)

            # print("Shapefile reprojected")
            # ---------------------------------------------------
            # STEP 3: Admin-level calculations
            # ---------------------------------------------------
            for _, admin_shape in intersected_shapes.iterrows():
                admin_geom = admin_shape.geometry
                admin_id = admin_shape["loc_id"]
                print(f"Processing admin id: {admin_id}")

                max_wind_speed = max_wind_by_loc.get(admin_id, np.nan)

                if not np.isfinite(max_wind_speed):
                    print(f"⚠️ Max wind speed is NaN for admin_id={admin_id} → skipping admin")
                    continue  # skip this admin entirely


                geom_pieces_cea, geom_pieces_wgs84 = split_antimeridian_geom(admin_geom)
                if not geom_pieces_cea:
                    geom_pieces_cea = [admin_geom]
                    geom_pieces_wgs84 = [gpd.GeoSeries([admin_geom], crs="ESRI:54034").to_crs("EPSG:4326").iloc[0]]

                # print(f"Split into {len(geom_pieces_cea)} pieces after antimeridian check")

                person_storm_hours_total = 0.0
                population_exposed_total = 0.0

                pop_sum, special_region_flag = get_population_total(
                    pop_df=pop_df,
                    year=year,
                    admin_id=admin_id,
                )

                for piece_cea, piece_wgs84 in zip(geom_pieces_cea, geom_pieces_wgs84):   
                    if basin != "NA":                 
                        piece_wgs84 = normalize_geom_to_0_360(piece_wgs84)
                    try:
                        admin_exposure = (
                            exposure_raster
                            .clip(piece_wgs84)
                            .mask(piece_wgs84, all_touched=True)
                        )
                    except rasterio.errors.WindowError:
                        # No overlap with raster → skip this piece
                        continue
                    
                    # take the intersection of piece_wgs84 with the raster bounds
                    minx, maxx, miny, maxy = admin_exposure.bounds
                    buffer = 0.2
                    minx -= buffer
                    maxx += buffer
                    miny -= buffer
                    maxy += buffer
                    raster_bounds = box(minx, miny, maxx, maxy)


                    # Intersect and clip
                    intersection = piece_wgs84.intersection(raster_bounds)
                    intersection_gdf = gpd.GeoDataFrame(geometry=[intersection], crs="EPSG:4326")
                    intersection_cea = intersection_gdf.to_crs("ESRI:54034").iloc[0].geometry
                    intersection_cea_bounds = intersection_cea.bounds
                    

                    # ----------------------------
                    # Load population
                    # ----------------------------
                    try:
                        pop_piece = load_in_gridded_population(year, 100, bounds=intersection_cea_bounds)
                        pop_piece_masked = pop_piece.mask(piece_cea, all_touched=True)
                    except rasterio.errors.WindowError:
                        print(f"[INFO] Admin piece {piece_cea} does not intersect population raster → skipping piece")
                        continue
                    except ValueError as e:
                        # Catch NaN window / invalid bounds
                        print(f"[INFO] Admin piece {piece_cea} failed to load population ({e}) → skipping piece")
                        continue
                    
                    # --- use masked population as the reference grid ---
                    pop_arr = pop_piece_masked._ndarray.astype(np.float32, copy=False)

                    # --- resample exposure to masked population grid ---
                    exposure_resampled = admin_exposure.resample_to(pop_piece_masked, resampling="nearest")

                    # optional but safer: enforce identical mask explicitly
                    exposure_resampled = exposure_resampled.mask(piece_cea, all_touched=True)

                    exposure_arr = exposure_resampled._ndarray.astype(np.float32, copy=False)


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
                        # print(f"Pop exposed: {population_exposed_total}")
                    # skip if valid mask is empty → no exposed population in this piece
                    else:
                        print("[INFO] No valid population-exposure cells in piece → skipping piece")
                        del pop_piece, pop_piece_masked, exposure_resampled
                        del pop_arr, exposure_arr
                        del pop_mask, exposure_mask, valid_mask
                        gc.collect()
                        continue
                        
                    # cleanup piece memory
                    del pop_piece, pop_piece_masked, exposure_resampled
                    del pop_arr, exposure_arr
                    del pop_mask, exposure_mask, valid_mask
                    gc.collect()

                
                storm_records.append(
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


            # save storm-level exposure immediately
            if storm_records:
                storm_df = pd.DataFrame.from_records(storm_records)
                save_storm_exposure(
                    exposure_df=storm_df,
                    source_id=source_id,
                    variant_label=variant_label,
                    experiment_id=experiment_id,
                    batch_year=batch_year,
                    basin=basin,
                    year=year,
                    draw=draw,
                    storm_id=storm_id,
                )
                del storm_df
            # ---------------------------------------------------
            # storm cleanup
            # ---------------------------------------------------

            del intensity_raster
            del exposure_raster
            del intersected_shapes
            del storm_records
            

            gc.collect()

        # Save yearly exposure after processing all storms in the year
        save_yearly_exposure(
            source_id=source_id,
            variant_label=variant_label,
            experiment_id=experiment_id,
            batch_year=batch_year,
            basin=basin,
            year=year,
            draw=draw,
        )

    # save draw-level exposure after processing all years in the draw
    save_draw_dataframe(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    )

    return 



def main(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw_batch: str,
    num_cores: int,
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

    run_parallel(
        runner=process_single_draw,
        arg_list=draw_args,
        num_cores=num_cores,
    )
    print("Completed Parallel Tasks")

    for draw in draws:
        draw_dir = (
            SAVE_ROOT
            / source_id
            / variant_label
            / experiment_id
            / batch_year
            / basin
            / f"tc_risk_draw_{draw}"
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
    num_cores=num_cores,
)
    