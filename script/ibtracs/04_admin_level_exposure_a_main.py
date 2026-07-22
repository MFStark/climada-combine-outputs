import os
import argparse
import traceback
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import xarray as xr  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import rasterra as rt  # type: ignore
import rasterio  # type: ignore
import shapely
import dask.array as da  # type: ignore

from rasterra import RasterArray  # type: ignore
from affine import Affine  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
from rasterio.features import shapes  # type: ignore
from rra_tools.parallel import run_parallel  # type: ignore
from shapely.geometry import box, shape, Polygon, MultiPolygon, GeometryCollection, LineString
from shapely.ops import split, unary_union
import pyarrow.parquet as pq  # type: ignore
from rasterra import RasterArray  # type: ignore

warnings.simplefilter("ignore", FutureWarning)


parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--year", type=str, required=True, help="Batch Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--admin_level", type=int, required=True, help="Admin level (e.g., 0, 1 or 2)")



# Parse arguments
args = parser.parse_args()
year = args.year
basin = args.basin
admin_level = args.admin_level





# Constants
ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage1")
SAVE_ROOT = Path(f"/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage4a_metadata_admin{admin_level}")

GDF_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/")
GRIDED_POP_PATH = Path("/mnt/team/rapidresponse/pub/population-model/results/2026_05_16/")
POP_TOTALS_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2023_all_years.parquet")
ANTIMERIDIAN = LineString([(180, -90), (180, 90)])
SHP_ROOT_NORMALIZED_HIGHER = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29')
SHP_PATH_NORMALIZED_A0 = Path('/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0_normalized.parquet')

SOURCE_ID = "ibtracs"
VARIANT_LABEL = "official"
EXPERIMENT_ID = "historical"


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

def reproject_shapefile_to_equal_area(intersected_shapes):
    # --- Normalize longitudes if crossing antimeridian ---
    maxx = intersected_shapes.geometry.bounds.maxx.max()
    if maxx > 180:
        intersected_shapes = normalize_longitudes(intersected_shapes)

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


def generate_basin_template_raster(
    year: int | str,
    basin: str,
    res: float = 0.1,
):
    # read in summary extent of all storms in the basin-year
    summary_file = (
        ROOT
        / "ibtracs"
        / "official"
        / "historical"
        / str(year)
        / basin
        / "storm_extent_summary.csv"
    )

    if not summary_file.exists():
        raise FileNotFoundError(f"Storm extent summary not found: {summary_file}")
    
    extent_df = pd.read_csv(summary_file)

    # get min/max lat/lon across all storms
    lon_min = extent_df["lon_min"].min()
    lon_max = extent_df["lon_max"].max()
    lat_min = extent_df["lat_min"].min()
    lat_max = extent_df["lat_max"].max()

    # generate template raster
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
    raster = RasterArray(data=data,
                         transform=transform,
                         crs="EPSG:4326",
                         no_data_value=np.nan
                         )
    return raster

##############################
#     Load Raw PAF Raster    #
##############################

def get_metric_dir(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    metric: str,
    root_dir: Path = ROOT,
):
    year = str(year)

    root_dir = root_dir / source_id / variant_label / experiment_id / year / basin / metric

    return root_dir



##########################################
#           Load Shapefile               #
##########################################

def load_shapefiles(admin_level: int = 0):
    file_name = f"global_WGS84_admin{admin_level}.parquet"
    shapefile=gpd.read_parquet(GDF_ROOT / file_name)

    return shapefile


def load_shapefiles_normalized(admin_level: int = 0) -> gpd.GeoDataFrame:
    """Load antimeridian-normalized admin shapes (parquet for admin 0, .shp otherwise)."""
    if admin_level == 0:
        return gpd.read_parquet(SHP_PATH_NORMALIZED_A0)
    shp_path = SHP_ROOT_NORMALIZED_HIGHER / f"lbd_standard_admin_{admin_level}.shp"
    return gpd.read_file(shp_path)
##########################################
#           Load Population              #
##########################################
def load_population_dataframe():
    pop_df = pd.read_parquet(POP_TOTALS_PATH)

    return pop_df

def get_population_total(pop_df: pd.DataFrame, year: str | int, admin_id: int):

    subset = pop_df[
        (pop_df["year_id"] == int(year)) &
        (pop_df["location_id"] == admin_id)
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
    pop_path = GRIDED_POP_PATH
    
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

def save_storm_metadata(
    meta_df: pd.DataFrame,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: int | str,
    basin: str,
    storm_id: str | int,
    save_root: Path = SAVE_ROOT,
):
    save_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / str(year)
        / basin
        / "storm_metadata"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"storm_{storm_id}_{basin}_{source_id}_{variant_label}_{experiment_id}_{year}.parquet"

    save_path = save_dir / file_name

    meta_df.to_parquet(save_path, index=False)
    os.chmod(save_path, 0o775)
    os.chmod(save_dir, 0o775)


def save_yearly_exposure(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    save_root: Path = SAVE_ROOT,
):
    save_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / str(year)
        / basin
        / "yearly_metadata"
    )
    save_dir.mkdir(parents=True, exist_ok=True)


    file_name = f"metadata_{year}_{basin}_{source_id}_{variant_label}_{experiment_id}.parquet"

    save_path = save_dir / file_name

    storm_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / str(year)
        / basin
        / "storm_metadata"
    )

    pattern = f"*_{basin}_{source_id}_{variant_label}_{experiment_id}_{year}.parquet"
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
    os.chmod(save_path, 0o775)
    os.chmod(save_dir, 0o775)


################################
# Check Completion Functions   #
################################

def check_if_storm_is_complete(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
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
        / str(year)
        / basin
        / "storm_metadata"
    )

    filename = f"storm_{storm_id}_{basin}_{source_id}_{variant_label}_{experiment_id}_{year}.parquet"
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
    year: str | int,
    basin: str,
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
        / str(year)
        / basin
        / "yearly_metadata"
    )

    file_name = f"metadata_{year}_{basin}_{source_id}_{variant_label}_{experiment_id}.parquet"

    save_path = save_dir / file_name

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


###########################################
#                Main                     #
###########################################
import time

def process_ibtracs_year(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    admin_level: int,
    save_root: Path = SAVE_ROOT,
):

    intensity_dir = get_metric_dir(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        year=year,
        basin=basin,
        metric="intensity",
    )

    exposure_dir = get_metric_dir(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        year=year,
        basin=basin,
        metric="exposure_hours",
    )

    # CHECK IF INTENSITY DATA EXISTS
    nc_files = list(intensity_dir.glob("*.nc")) if intensity_dir.exists() else []

    if not nc_files:
        print(f"⚠️ No intensity .nc files found for basin {basin}, year {year}. Returning empty rasters.")
        return

    if basin == "NA":
        shapefile = load_shapefiles_normalized(admin_level=admin_level)
    else:
        shapefile = load_shapefiles(admin_level=admin_level)

    template_raster = generate_basin_template_raster(year, basin, res=0.1)

    storm_paths_in_year = [str(f) for f in nc_files]
    print(f"Number of storms: {len(storm_paths_in_year)}")

    for storm_path in storm_paths_in_year:
        start_time = time.time()
        storm_records = []

        storm_ds = xr.open_dataset(storm_path, decode_timedelta=False, chunks="auto")
        storm_id = storm_ds.attrs.get("storm_id", Path(storm_path).stem)
        # print(f"Processing storm id: {storm_id}")

        # check if storm is already complete
        if check_if_storm_is_complete(
            source_id=source_id,
            variant_label=variant_label,
            experiment_id=experiment_id,
            year=year,
            basin=basin,
            storm_id=storm_id,
            save_root=save_root,
        ):
            print(f"Storm {storm_id} already processed → skipping")
            continue

        storm_exposure_path = exposure_dir / f"{storm_id}.nc"
        storm_exposure = xr.open_dataset(storm_exposure_path, decode_timedelta=False, chunks="auto")

        # Normalize longitudes if basin is NA
        if basin == "NA":
            storm_ds = normalize_dataset(storm_ds)
            storm_exposure = normalize_dataset(storm_exposure)

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
        
        intensity_raster._ndarray = intensity_raster._ndarray
        exposure_raster._ndarray = exposure_raster._ndarray
        # print(f"Computed raster data")

        try: 
            intensity_raster = clean_raster(intensity_raster)
        except ValueError:
            print("No affected pixels found in intensity raster → skipping storm")

        try:
            exposure_raster = clean_raster(exposure_raster).resample_to(target=intensity_raster, resampling="nearest") # ensures same grid after clean
        except ValueError:
            print("No affected pixels found in exposure raster → skipping storm")

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
            continue

        intersected_shapes = reproject_shapefile_to_equal_area(intersected_shapes)


        for admin_shape in intersected_shapes.itertuples(index=False):
            admin_geom = admin_shape.geometry
            admin_id = admin_shape.loc_id
            print(f"Processing admin id: {admin_id}")

            # Per-admin try/except so one bad polygon doesn't kill the
            # storm (which would also skip the per-storm save below).
            try:
                geom_pieces_cea, geom_pieces_wgs84 = split_antimeridian_geom(admin_geom)
                if not geom_pieces_cea:
                    geom_pieces_cea = [admin_geom]
                    geom_pieces_wgs84 = [gpd.GeoSeries([admin_geom], crs="ESRI:54034").to_crs("EPSG:4326").iloc[0]]

                print(f"Split into {len(geom_pieces_cea)} pieces after antimeridian check")

                total_1km = 0
                affected_1km = 0
                pop_affected = 0
                area_km2_list = []
                minx_list, maxx_list, miny_list, maxy_list = [], [], [], []

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
                        pop_piece = load_in_gridded_population(year, 1000, bounds=intersection_cea_bounds)
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

                    # if pop arr is entirely NaN or zero, skip
                    if not np.isfinite(pop_arr).any() or (pop_arr > 0).sum() == 0:
                        print(f"[INFO] No valid population pixels in piece → skipping")
                        continue

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

                    # check if valid population exists in the piece, if not skip
                    if not valid_mask.any():
                        print(f"[INFO] No valid population-exposure pixels in piece → skipping")
                        continue

                    # get bounds of the masked population piece
                    pop_xmin, pop_xmax, pop_ymin, pop_ymax = pop_piece_masked.bounds
                    minx_list.append(pop_xmin)
                    maxx_list.append(pop_xmax)
                    miny_list.append(pop_ymin)
                    maxy_list.append(pop_ymax)

                    pop_bounds = pop_piece_masked.bounds
                    minx, maxx, miny, maxy = pop_bounds
                    area = (maxx - minx) * (maxy - miny)
                    area_km2 = area / 1e6
                    area_km2_list.append(area_km2)

                    total_1km += pop_arr.size
                    affected_1km += valid_mask.sum()
                    pop_affected += pop_arr[valid_mask].sum()

                total_100m = total_1km * 100

                if total_1km == 0:
                    print(f"[WARNING] Total population pixels at 1km resolution is zero for admin {admin_id} → skipping percentage calculation")
                    percent_affected = np.nan
                else:
                    percent_affected = (affected_1km / total_1km) * 100

                affected_100m = affected_1km * 100
                area_100m2 = sum(area_km2_list) * 1e6
                end_time = time.time()
                elapsed_time = end_time - start_time
                elapsed_time_str = f"{elapsed_time:.2f}"

                # check if storm is valid (has affected population) before saving
                if pop_affected == 0:
                    print(f"[INFO] No affected population for admin {admin_id} → skipping storm record")
                    continue

                if affected_1km == 0:
                    print(f"[INFO] No affected population pixels at 1km resolution for admin {admin_id} → skipping storm record")
                    continue

                storm_records.append({
                    "source_id": source_id,
                    "basin": basin,
                    "storm_id": storm_id,
                    "year": year,
                    "location_id": admin_id,
                    "projection": pop_piece.crs,
                    "resolution": "estimated 100m from 1km",
                    "total_population_pixels_100m": total_100m,
                    "affected_population_pixels_100m": affected_100m,
                    "percent_affected_100m": percent_affected,
                    "population_affected": pop_affected,
                    "xmin": min(minx_list),
                    "xmax": max(maxx_list),
                    "ymin": min(miny_list),
                    "ymax": max(maxy_list),
                    "area_100m2": area_100m2,
                    "processing_time_seconds": elapsed_time_str,
                })
            except Exception as e:
                print(
                    f"❌ Admin {admin_id} failed for storm {storm_id} in basin "
                    f"{basin}, year {year}: {type(e).__name__}: {e}"
                )
                traceback.print_exc()
                continue

        storm_df = pd.DataFrame(storm_records)

        save_storm_metadata(
            meta_df=storm_df,
            source_id=source_id,
            variant_label=variant_label,
            experiment_id=experiment_id,
            year=year,
            basin=basin,
            storm_id=storm_id,
            save_root=save_root,
        )

    # save yearly metadata
    save_yearly_exposure(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        year=year,
        basin=basin,
        save_root=save_root,
    )


process_ibtracs_year(
    source_id=SOURCE_ID,
    variant_label=VARIANT_LABEL,
    experiment_id=EXPERIMENT_ID,
    year=args.year,
    basin=args.basin,
    admin_level=args.admin_level,
    save_root=SAVE_ROOT,
)