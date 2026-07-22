from pathlib import Path
from typing import NamedTuple
import xarray as xr  # type: ignore
import numpy as np
import rasterra as rt  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
from affine import Affine  # type: ignore
import os
import warnings
import argparse
import traceback
from rasterio.features import shapes, rasterize  # type: ignore
from rasterio.errors import WindowError  # type: ignore
from rasterra import RasterArray  # type: ignore
import shapely
from shapely.geometry import (
    box,
    shape,
    Polygon,
    MultiPolygon,
    GeometryCollection,
    LineString,
)
from shapely.ops import split, unary_union
import re
from rasterra import RasterArray  # type: ignore

warnings.simplefilter("ignore", FutureWarning)


parser = argparse.ArgumentParser(description="Run CLIMADA stage 4B for a group of tasks")

parser.add_argument(
    "--grouped_tasks_parquet", type=str, required=True,
    help="Path to the per-workflow grouped task parquet produced by the launcher",
)
parser.add_argument(
    "--group_id", type=int, required=True,
    help="Integer group_id; selects the subset of rows to process from the parquet",
)
parser.add_argument(
    "--admin_level", type=int, required=False, default=0, choices=[0, 1],
    help="Admin level (0 or 1)",
)

args = parser.parse_args()
grouped_tasks_parquet = args.grouped_tasks_parquet
group_id = args.group_id
ADMIN_LEVEL = args.admin_level


# Constants
ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1_v2")
GRIDED_POP_PATH = Path("/mnt/team/rapidresponse/pub/population-model/results/2026_05_16/")
POP_TOTALS_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2023_all_years.parquet")

_SHP_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile")

if ADMIN_LEVEL == 0:
    SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b_v2/")
    GDF_PATH = _SHP_ROOT / "global_WGS84_admin0.parquet"
    SHP_ROOT_NORMALIZED = _SHP_ROOT / "global_WGS84_admin0_normalized.parquet"
elif ADMIN_LEVEL == 1:
    SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b_v2_admin1/")
    GDF_PATH = _SHP_ROOT / "global_WGS84_admin1_0_360.parquet"
    SHP_ROOT_NORMALIZED = _SHP_ROOT / "global_WGS84_admin1_0_360.parquet"
else:
    raise ValueError(f"admin_level must be 0 or 1, got {ADMIN_LEVEL!r}")

# Antimeridian line — defined in WGS84
ANTIMERIDIAN = shapely.geometry.LineString([(180, -90), (180, 90)])
CEA_MAX_X    = 20_037_508.0
CEA_MIN_X    = -20_037_508.0

class StormMeta(NamedTuple):  # type: ignore
    storm_path: Path
    start_year: int
    end_year: int
    storm_id: str


##############################
#     Helper Functions       #
##############################
def ensure_min_grid(
    da: "xr.DataArray | xr.Dataset",
    buffer_deg: float = 0.1,
) -> "xr.DataArray | xr.Dataset":
    """
    Pad a single-pixel lat/lon dimension with NaN neighbors so rasterization
    sees at least a 3-cell axis. Works for both DataArray and Dataset.
    """
    if da.lat.size == 1:
        c = float(da.lat.values[0])
        da = da.reindex(lat=[c-buffer_deg, c, c+buffer_deg], fill_value=np.nan)

    if da.lon.size == 1:
        c = float(da.lon.values[0])
        da = da.reindex(lon=[c-buffer_deg, c, c+buffer_deg], fill_value=np.nan)

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
    a, _b, c, _d, e, f = transform[:6]

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


def generate_storm_template_raster(storm_ds: xr.Dataset, res: float = 0.1, buffer_deg: float = 1.0) -> RasterArray:
    """
    Build a minimal 0.1° template raster from the storm's lat/lon extent.
    Avoids allocating a full-basin array just to normalize grid resolution.
    """
    lat_vals = storm_ds.lat.values
    lon_vals = storm_ds.lon.values

    lat_min = np.floor((float(lat_vals.min()) - buffer_deg) / res) * res
    lat_max = np.ceil((float(lat_vals.max()) + buffer_deg) / res) * res
    lon_min = np.floor((float(lon_vals.min()) - buffer_deg) / res) * res
    lon_max = np.ceil((float(lon_vals.max()) + buffer_deg) / res) * res

    n_cols = int(round((lon_max - lon_min) / res))
    n_rows = int(round((lat_max - lat_min) / res))

    transform = Affine(res, 0, lon_min, 0, -res, lat_max)
    return RasterArray(
        data=np.zeros((n_rows, n_cols), dtype=np.float32),
        transform=transform,
        crs="EPSG:4326",
        no_data_value=np.nan,
    )


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

def get_draw_zarr_path(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int | str,
    metric: str,
) -> Path:
    """
    Locate draw-level storm Zarr store produced by Stage 1.
    """
    start_year, end_year = batch_year.split("-")
    draw_text = "" if draw == 0 else f"_e{int(draw) - 1}"

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

def load_shapefiles() -> gpd.GeoDataFrame:
    return gpd.read_parquet(GDF_PATH)


def load_shapefiles_normalized() -> gpd.GeoDataFrame:
    """Load NA-basin admin shapes in -180..180 lon convention.
    Admin0: pre-normalized parquet with special regions merged.
    Admin1: FHS-filtered 0-360 parquet, normalized to -180..180 at load time."""
    gdf = gpd.read_parquet(SHP_ROOT_NORMALIZED)
    if ADMIN_LEVEL == 1:
        gdf = normalize_longitudes(gdf)
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
    """
    Split a CEA geometry that crosses the antimeridian into valid sub-pieces.

    Returns
    -------
    out_cea   : list of Shapely geometries in ESRI:54034
    out_wgs84 : list of Shapely geometries in EPSG:4326
    """

    # -----------------------------------------------
    # STEP 1: Convert to WGS84 for antimeridian check
    # -----------------------------------------------
    gdf    = gpd.GeoSeries([geom_cea], crs="ESRI:54034")
    geom_ll = gdf.to_crs("EPSG:4326").iloc[0]
    minx, miny, maxx, maxy = geom_ll.bounds

    # -----------------------------------------------
    # STEP 2: Check if splitting is needed
    # Use a tighter heuristic — also check CEA bounds
    # directly to catch boundary-adjacent geometries
    # -----------------------------------------------
    cea_minx, _, cea_maxx, _ = geom_cea.bounds
    near_cea_boundary = (cea_maxx > CEA_MAX_X * 0.90 or
                         cea_minx < CEA_MIN_X * 0.90)
    crosses_antimeridian = (maxx > 180 or minx < -180 or
                            (maxx - minx) > 180)

    if not crosses_antimeridian and not near_cea_boundary:
        return [geom_cea], [geom_ll]

    # -----------------------------------------------
    # STEP 3: Split on antimeridian in WGS84
    # -----------------------------------------------
    try:
        result = split(geom_ll, ANTIMERIDIAN)
        pieces = [
            g for g in result.geoms
            if g.geom_type in ("Polygon", "MultiPolygon")
        ]
    except Exception as e:
        print(f"[WARN] split_antimeridian_geom split failed ({e}) → returning original")
        return [geom_cea], [geom_ll]

    if not pieces:
        return [geom_cea], [geom_ll]

    # -----------------------------------------------
    # STEP 4: Classify pieces using bounds midpoint
    #         not centroid — more robust for island geometries
    #         near ±180°
    # -----------------------------------------------
    west, east = [], []
    for g in pieces:
        gminx, _, gmaxx, _ = g.bounds
        midx = (gminx + gmaxx) / 2
        if midx < 0:
            west.append(g)
        else:
            east.append(g)

    west_union = unary_union(west) if west else None
    east_union = unary_union(east) if east else None
    out_ll = [g for g in [west_union, east_union] if g is not None]

    # -----------------------------------------------
    # STEP 5: Reproject pieces to CEA individually
    #         then validate — reject any piece whose
    #         CEA width spans >25% of the full projection
    #         width (wrap-around artifact)
    # -----------------------------------------------
    CEA_WRAP_THRESHOLD = CEA_MAX_X * 0.50  # 50% of half-width = ~10,000km

    out_cea   = []
    out_wgs84 = []

    for geom_wgs84 in out_ll:
        try:
            geom_reprojected = (
                gpd.GeoSeries([geom_wgs84], crs="EPSG:4326")
                .to_crs("ESRI:54034")
                .iloc[0]
            )
        except Exception as e:
            print(f"[WARN] Reprojection failed for piece ({e}) → skipping piece")
            continue

        piece_minx, _, piece_maxx, _ = geom_reprojected.bounds
        piece_width = piece_maxx - piece_minx

        if piece_width > CEA_WRAP_THRESHOLD:
            print(
                f"[WARN] Rejecting wrap-around CEA piece: "
                f"width={piece_width/1000:.0f}km "
                f"(threshold={CEA_WRAP_THRESHOLD/1000:.0f}km) — "
                f"antimeridian artifact, skipping"
            )
            continue

        # Additional guard — clamp bounds that sit right at the CEA edge
        if piece_maxx > CEA_MAX_X or piece_minx < CEA_MIN_X:
            print(
                f"[WARN] CEA piece exceeds projection bounds — clamping: "
                f"xmin={piece_minx:.0f} xmax={piece_maxx:.0f}"
            )

        out_cea.append(geom_reprojected)
        out_wgs84.append(geom_wgs84)

    # -----------------------------------------------
    # STEP 6: Fallback — if all pieces rejected,
    #         return original geometry with a warning
    # -----------------------------------------------
    if not out_cea:
        print(
            f"[WARN] All split pieces rejected for geometry with bounds "
            f"({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}) → "
            f"returning original unsplit geometry"
        )
        return [geom_cea], [geom_ll]

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
    location_id: str | int,
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

    file_name = f"storm_{storm_id}_loc_{location_id}_{basin}_{source_id}_{variant_label}_{experiment_id}_{start_year}01_{end_year}12{draw_text}.parquet"

    save_path = save_dir / file_name

    exposure_df.to_parquet(save_path, index=False)

    # Set permissions on the file we just wrote and its parent so the group
    # can read + traverse. Wrapped because a chmod failure shouldn't
    # invalidate a successful parquet write.
    try:
        os.chmod(save_path, 0o775)
        os.chmod(save_path.parent, 0o775)
    except Exception as e:
        print(f"⚠️ Could not set permissions for {save_path}: {e}")


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

    try:
        os.chmod(save_path, 0o775)
        os.chmod(save_path.parent, 0o775)
    except Exception as e:
        print(f"⚠️ Could not set permissions for {save_path}: {e}")


def _is_task_complete(row, save_root: Path = SAVE_ROOT) -> bool:
    """Return True if a ≥1 KB output parquet already exists for this task."""
    draw_dir = (
        save_root
        / row.source_id / row.variant_label / row.experiment_id
        / row.batch_year / row.basin
        / f"tc_risk_draw_{int(row.draw)}" / "storm_exposure"
    )
    if not draw_dir.exists():
        return False
    pattern = f"storm_{int(row.storm_id):04d}_loc_{int(row.location_id)}_*.parquet"
    return any(f.stat().st_size >= 1024 for f in draw_dir.rglob(pattern))


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

    try:
        os.chmod(save_path, 0o775)
        os.chmod(save_path.parent, 0o775)
    except Exception as e:
        print(f"⚠️ Could not set permissions for {save_path}: {e}")


###########################################
#                Main                     #
###########################################
def process_single_storm(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int | str,
    storm_id: str,
    location_id: int | str,
    num_cores: int = 1,
    ):

    draw = int(draw)  # ensure draw is int for path construction
    location_id = int(location_id)  # ensure location_id is int for shapefile filtering
    
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

    storm_records = []
    storm_path = intensity_draw_store / f"storm_{int(storm_id):04d}"

    storm_ds = xr.open_zarr(storm_path, consolidated=False, chunks="auto")
    start_date = storm_ds.attrs.get("start_date", "unknown")
    year = pd.to_datetime(start_date).year if start_date != "unknown" else "unknown"

    # ---------------------------------------------------
    # Load exposure
    # ---------------------------------------------------
    try:
        storm_exposure = get_exposure_storm_from_draw(
            exposure_draw_store,
            storm_id,
        )
    except (FileNotFoundError, KeyError):
        storm_ds.close()
        return

    # Load large reference data only after confirming this storm needs processing
    if basin == "NA":
        shapefile = load_shapefiles_normalized()
    else:
        shapefile = load_shapefiles()
    pop_df = load_population_dataframe()

    try:
        shapefile = shapefile[shapefile["loc_id"] == location_id].copy()
    except KeyError:
        print(f"Location ID {location_id} not found in shapefile → skipping storm")
        storm_ds.close()
        storm_exposure.close()
        del storm_ds, storm_exposure
        return

    # Normalize longitudes if basin is NA
    if basin == "NA":
        storm_ds = normalize_dataset(storm_ds)
        storm_exposure = normalize_dataset(storm_exposure)

    # Generate a minimal template raster based on storm extent to optimize rasterization
    template_raster = generate_storm_template_raster(storm_ds, res=0.1)

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
    del template_raster

    # Clean up
    storm_ds.close()
    storm_exposure.close()
    del storm_exposure
    del storm_ds

    try: 
        intensity_raster = clean_raster(intensity_raster)
    except ValueError:
        print("No affected pixels found in intensity raster → skipping storm")
        return
    try:
        exposure_raster = clean_raster(exposure_raster)
    except ValueError:
        print("No affected pixels found in exposure raster → skipping storm")
        del intensity_raster, exposure_raster
        return

    # ---------------------------------------------------
    # Subset admin shapes intersecting storm
    # ---------------------------------------------------
    intersected_shapes = intersect_shapefile_with_rasters(
        shapefile,
        intensity_raster,
        exposure_raster,
        buffer_degrees=0.0,
    )
    del shapefile

    if intersected_shapes.empty:
        print("No intersecting shapes for storm")
        del intensity_raster, exposure_raster
        del intersected_shapes
        return
    
    # ---------------------------------------------------
    # Vectorized max windspeed across all intersecting admins
    # ---------------------------------------------------
    admin_ids_list = intersected_shapes["loc_id"].tolist()
    id_to_idx = {aid: i + 1 for i, aid in enumerate(admin_ids_list)}
    idx_to_id = {i + 1: aid for i, aid in enumerate(admin_ids_list)}

    admin_raster = rasterize(
        shapes=[(geom, id_to_idx[aid]) for geom, aid in zip(intersected_shapes.geometry, admin_ids_list)],
        out_shape=intensity_raster._ndarray.shape,
        transform=intensity_raster.transform,
        fill=0,
        dtype=np.int32,
        all_touched=True,
    )
    intensity_arr = intensity_raster._ndarray

    max_wind_by_loc = {}

    for idx, admin_id in idx_to_id.items():
        vals = intensity_arr[admin_raster == idx]
        finite_vals = vals[np.isfinite(vals)]
        max_wind_by_loc[admin_id] = float(finite_vals.max()) if len(finite_vals) > 0 else np.nan

    del admin_raster, intensity_arr

    intersected_shapes = reproject_shapefile_to_equal_area(intersected_shapes)
    del intensity_raster
    # ---------------------------------------------------
    # STEP 3: Admin-level calculations
    # ---------------------------------------------------
    for admin_shape in intersected_shapes.itertuples(index=False):
        admin_id = getattr(admin_shape, "loc_id", None)
        try:
            admin_geom = admin_shape.geometry

            max_wind_speed = max_wind_by_loc.get(admin_id, np.nan)

            if not np.isfinite(max_wind_speed):
                print(f"⚠️ Max wind speed is NaN for admin_id={admin_id} → skipping admin")
                continue  # skip this admin entirely


            geom_pieces_cea, geom_pieces_wgs84 = split_antimeridian_geom(admin_geom)
            if not geom_pieces_cea:
                geom_pieces_cea = [admin_geom]
                geom_pieces_wgs84 = [gpd.GeoSeries([admin_geom], crs="ESRI:54034").to_crs("EPSG:4326").iloc[0]]

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
                except WindowError:
                    continue

                # ----------------------------
                # Guard: zero-dimension raster check
                # Boundary-adjacent pieces can produce empty arrays
                # after clip — catch here before any downstream use
                # ----------------------------
                exp_h, exp_w = admin_exposure._ndarray.shape
                if exp_h == 0 or exp_w == 0:
                    print(
                        f"[INFO] Zero-dimension admin_exposure for "
                        f"storm={storm_id} admin={admin_id} basin={basin} "
                        f"({exp_h} x {exp_w}) → skipping piece"
                    )
                    del admin_exposure
                    continue

                # Clip piece to the raster extent (plus buffer) before reprojecting
                minx, maxx, miny, maxy = admin_exposure.bounds
                buffer = 0.2
                minx -= buffer
                maxx += buffer
                miny -= buffer
                maxy += buffer
                raster_bounds = box(minx, miny, maxx, maxy)

                intersection = piece_wgs84.intersection(raster_bounds)
                intersection_cea_bounds = (
                    gpd.GeoSeries([intersection], crs="EPSG:4326")
                    .to_crs("ESRI:54034")
                    .iloc[0]
                    .bounds
                )

                # ----------------------------
                # Load population
                # ----------------------------
                try:
                    pop_piece = load_in_gridded_population(year, 100, bounds=intersection_cea_bounds)
                    pop_piece_masked = pop_piece.mask(piece_cea, all_touched=True)
                    del pop_piece  # free unmasked copy; pop_piece_masked is sufficient
                except WindowError:
                    print(f"[INFO] Admin piece does not intersect population raster → skipping piece")
                    del admin_exposure
                    continue
                except ValueError as e:
                    print(f"[INFO] Admin piece failed to load population ({e}) → skipping piece")
                    del admin_exposure
                    continue

                pop_arr = pop_piece_masked._ndarray.astype(np.float32, copy=False)
                exposure_resampled = admin_exposure.resample_to(pop_piece_masked, resampling="nearest")
                del admin_exposure  # no longer needed; exposure_resampled holds the resampled data
                exposure_resampled = exposure_resampled.mask(piece_cea, all_touched=True)
                exposure_arr = exposure_resampled._ndarray.astype(np.float32, copy=False)

                # Single boolean array instead of three to reduce peak by ~2x bool-array footprint
                valid_mask = (
                    (pop_arr > 0) & np.isfinite(pop_arr) &
                    (exposure_arr > 0) & np.isfinite(exposure_arr)
                )

                if valid_mask.any():
                    person_storm_hours_total += (pop_arr[valid_mask] * exposure_arr[valid_mask]).sum()
                    population_exposed_total += pop_arr[valid_mask].sum()
                else:
                    print("[INFO] No valid population-exposure cells in piece → skipping piece")

                del pop_piece_masked, exposure_resampled
                del pop_arr, exposure_arr, valid_mask


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
        except Exception as e:
            print(
                f"❌ Admin {admin_id} failed for storm {storm_id} "
                f"({source_id}/{variant_label}/{experiment_id}/{batch_year}/"
                f"{basin}, draw {draw}): {type(e).__name__}: {e}"
            )
            traceback.print_exc()
            continue

    # Free large objects that are no longer needed before saving
    del pop_df, max_wind_by_loc, intersected_shapes, exposure_raster

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
            location_id=location_id,
        )
        del storm_df

    del storm_records


def _write_sentinel(group_hash: str) -> None:
    """Write a completion sentinel for this group. Called only after all tasks
    in the group have been processed without raising an unhandled exception."""
    sentinel_dir = SAVE_ROOT / "_sentinels"
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = sentinel_dir / f"{group_hash}.done"
    sentinel_path.touch()
    try:
        import os
        os.chmod(sentinel_path, 0o775)
    except Exception:
        pass


def main(grouped_tasks_parquet: str, group_id: int) -> None:
    group_df = pd.read_parquet(grouped_tasks_parquet)
    group_df = group_df[group_df["group_id"] == group_id].reset_index(drop=True)

    if group_df.empty:
        print(f"⚠️ No tasks found for group_id={group_id} in {grouped_tasks_parquet}")
        return

    total = len(group_df)
    group_hash = group_df["group_hash"].iloc[0]
    print(f"Group {group_id} (hash={group_hash[:8]}…): {total} tasks.")

    for i, row in enumerate(group_df.itertuples(index=False)):
        print(
            f"  [{i + 1}/{total}] "
            f"storm={row.storm_id} loc={row.location_id} "
            f"draw={row.draw} basin={row.basin} batch={row.batch_year}"
        )
        process_single_storm(
            source_id=row.source_id,
            variant_label=row.variant_label,
            experiment_id=row.experiment_id,
            batch_year=row.batch_year,
            basin=row.basin,
            draw=row.draw,
            storm_id=row.storm_id,
            location_id=row.location_id,
            num_cores=1,
        )

    # Write sentinel only after all tasks complete without crashing.
    # If the worker crashes mid-group, no sentinel is written and the group
    # is resubmitted on the next launcher run.
    _write_sentinel(group_hash)
    print(f"  ✅ Group {group_id} complete — sentinel written.")


main(grouped_tasks_parquet, group_id)
    