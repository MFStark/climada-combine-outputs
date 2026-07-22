import os
import argparse
import traceback
from pathlib import Path

import xarray as xr  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import rasterra as rt  # type: ignore
import rasterio  # type: ignore
import shapely

from rasterio.features import shapes  # type: ignore
from rra_tools.parallel import run_parallel  # type: ignore
from shapely.geometry import box, shape, Polygon, MultiPolygon, GeometryCollection, LineString
from shapely.ops import split, unary_union
import pyarrow.parquet as pq  # type: ignore
from rasterra import RasterArray  # type: ignore

parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--year", type=str, required=True, help="Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--relative_risk", type=str, required=True, help="Relative risk type")
parser.add_argument("--admin_level", type=int, required=False, default=0, choices=[0, 1], help="Admin level (0 or 1)")

# Parse arguments
args = parser.parse_args()
year = args.year
basin = args.basin
relative_risk = args.relative_risk
admin_level = args.admin_level

# Constants
PAF_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage2/")
GRIDED_POP_PATH = Path("/mnt/team/rapidresponse/pub/population-model/results/2026_05_16/")
POP_TOTALS_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2023_all_years.parquet")
ANTIMERIDIAN = LineString([(180, -90), (180, 90)])

_SHP_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile")

if admin_level == 0:
    SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage3/")
    GDF_PATH = _SHP_ROOT / "global_WGS84_admin0.parquet"
    SHP_ROOT_NORMALIZED = _SHP_ROOT / "global_WGS84_admin0_normalized.parquet"
elif admin_level == 1:
    SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage3_admin1/")
    GDF_PATH = _SHP_ROOT / "global_WGS84_admin1_0_360.parquet"
    SHP_ROOT_NORMALIZED = _SHP_ROOT / "global_WGS84_admin1_0_360.parquet"
else:
    raise ValueError(f"admin_level must be 0 or 1, got {admin_level!r}")

SOURCE_ID = "ibtracs"
VARIANT_LABEL = "official"
EXPERIMENT_ID = "historical"
SAMPLE_NAME = "mean"
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

def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.assign_coords(
        lon=(((ds.lon + 180) % 360) - 180)
    ).sortby("lon")

    return ds

def subset_affected_area(
    rr_raster: rt.RasterArray,
    threshold: float = 0.0,
) -> rt.RasterArray:
    """
    Subset a RasterArray to the minimal bounding box where RR > threshold.

    If no pixels exceed the threshold, the original raster is returned.

    Parameters
    ----------
    rr_raster : RasterArray
        Storm relative risk raster.
    threshold : float
        Threshold defining affected pixels.

    Returns
    -------
    RasterArray
        Subset raster clipped to affected area, or original raster if none affected.
    """
    data = np.asarray(rr_raster.data)

    mask = np.isfinite(data) & (data > threshold)
    if not np.any(mask):
        # No affected pixels → return original raster
        print("[INFO] subset_affected_area: no pixels above threshold, returning original raster")
        return rr_raster

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

    # Clip raster to affected area
    return rr_raster.clip(gdf)

def reproject_shapefile_to_equal_area(intersected_shapes):
    # --- Normalize longitudes if crossing antimeridian ---
    maxx = intersected_shapes.geometry.bounds.maxx.max()
    if maxx > 180:
        intersected_shapes = normalize_longitudes(intersected_shapes)

    # Reproject shapes to equal-area CRS (cheap, vector-only).
    intersected_shapes = intersected_shapes.to_crs("ESRI:54034")

    intersected_shapes["geometry"] = intersected_shapes.geometry.apply(_polygonize)
    return intersected_shapes

##############################
#     Load Raw PAF Raster    #
##############################
def load_raw_paf_raster(
    source_id: str,
    variant_label: str,
    relative_risk: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    metric: str = "raw_paf",
    paf_root: Path = PAF_ROOT,
):
    year = str(year)
    root_dir = paf_root / source_id / variant_label / experiment_id  / year / basin / metric

    filename = (
        f"draw_mean_{metric}_{relative_risk}_{basin}_{source_id}_"
        f"{experiment_id}_{variant_label}_{year}.tif"
    )

    paf_path = root_dir / filename
    if not paf_path.exists():
        raise FileNotFoundError(f"PAF raster not found: {paf_path}")
    
    raw_paf_raster = rt.load_raster(paf_path)

    return raw_paf_raster

def clean_paf_raster(raw_paf_raster):

    # Ensure float32 and operate directly on raster array
    raw_paf_raster._ndarray = raw_paf_raster._ndarray.astype(np.float32, copy=False)

    # Convert 0 → NaN in-place
    raw_paf_raster._ndarray[raw_paf_raster._ndarray == 0] = np.nan

    raw_paf_raster = subset_affected_area(raw_paf_raster)

    return raw_paf_raster

##########################################
#           Load Shapefile               #
##########################################

def load_shapefiles():
    shapefile=gpd.read_parquet(GDF_PATH)

    return shapefile

def load_shapefiles_normalized(
    shapefile_root: Path = SHP_ROOT_NORMALIZED,
) -> gpd.GeoDataFrame:
    """Load NA-basin admin shapes in -180..180 lon convention.
    Admin0: pre-normalized parquet with special regions merged.
    Admin1: FHS-filtered 0-360 parquet, normalized to -180..180 at load time."""
    gdf = gpd.read_parquet(shapefile_root)
    if admin_level == 1:
        gdf = normalize_longitudes(gdf)
    return gdf

##########################################
#           Load Population              #
##########################################
def load_population_dataframe():
    pop_df = pd.read_parquet(POP_TOTALS_PATH)

    return pop_df

def get_population_total(pop_df: pd.DataFrame, year: int | str, admin_id: int):

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



##########################################
#            Save Functions              #
##########################################
def save_batch_paf_dataframe(
    paf_df: pd.DataFrame,
    source_id: str,
    variant_label: str,
    sample_name: str,
    relative_risk: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    save_root: Path = SAVE_ROOT,
):
    """
    Save batch-year population-weighted PAF dataframe.

    Expected paf_df columns (minimum):
        - location_id
        - year
        - total_population
        - population_weighted_paf
    """
    year = str(year)
    save_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / year
        / basin
        / "paf_df"
    )
    save_dir.mkdir(parents=True, exist_ok=True)


    filename = f"paf_{relative_risk}_{sample_name}_{basin}_{source_id}_{experiment_id}_{variant_label}_{year}.parquet"
    save_path = save_dir / filename

    # Optional: enforce consistent column order
    preferred_cols = [
        "storm_draw",
        "location_id",
        "year",
        "total_population",
        "population_weighted_paf",
    ]
    existing_cols = [c for c in preferred_cols if c in paf_df.columns]
    other_cols = [c for c in paf_df.columns if c not in existing_cols]
    paf_df = paf_df[existing_cols + other_cols]

    paf_df.to_parquet(save_path, index=False)
    os.chmod(save_path, 0o775)
    os.chmod(save_dir, 0o775)
    print(f"Saved batch-year PAF dataframe: {save_path}")


def check_if_year_complete(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    sample_name: str,
    relative_risk: str,
    experiment_id: str,
    batch_year: str,
    year: int,
    basin: str,
    save_root: Path = SAVE_ROOT,
) -> bool:
    """Return True if the yearly PAF Parquet exists and is valid.
    
    Returns False if missing, empty, zero-byte, or corrupt.
    """
    year = str(year)

    save_dir = (
        save_root
        / storm_draw
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / year
        / "paf_df"
    )

    start_year, end_year = batch_year.split("-")

    filename = (
        f"paf_{storm_draw}_{relative_risk}_{sample_name}_{basin}_"
        f"{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}.parquet"
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

def check_if_raw_paf_exists(
    source_id: str,
    variant_label: str,
    relative_risk: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    paf_root: Path = PAF_ROOT,
) -> bool:
    try:
        load_raw_paf_raster(
            source_id=source_id,
            variant_label=variant_label,
            relative_risk=relative_risk,
            experiment_id=experiment_id,
            year=year,
            basin=basin,
            paf_root=paf_root,
        )
        return True
    except FileNotFoundError:
        return False

##########################################
#            MAIN FUNCTION               #
##########################################
# import time
def process_single_year(
    year: str | int,
    basin: str,
    relative_risk: str,
    sample_name: str = SAMPLE_NAME,
):

    # load shapefile and population once per year
    if basin == "NA":
        shapefile = load_shapefiles_normalized()
    else:
        shapefile = load_shapefiles()
    pop_df = load_population_dataframe()
    paf_records = []

    # check if raw PAF raster exists before proceeding
    if not check_if_raw_paf_exists(
        source_id=SOURCE_ID,
        variant_label=VARIANT_LABEL,
        relative_risk=relative_risk,
        experiment_id=EXPERIMENT_ID,
        year=year,
        basin=basin,
    ):
        print(f"Raw PAF raster not found for year {year}, basin {basin}. Skipping.")
        return
    
    # load and clean PAF raster once
    raw_paf_raster = clean_paf_raster(
        load_raw_paf_raster(
            source_id=SOURCE_ID,
            variant_label=VARIANT_LABEL,
            relative_risk=relative_risk,
            experiment_id=EXPERIMENT_ID,
            year=year,
            basin=basin,
        )
    )


    intersected_shapes = intersect_shapefile_with_raster(
        shapefile,
        raw_paf_raster,
        buffer_degrees=0.0,
    )

    if len(intersected_shapes) == 0:
        print(f"No intersected shapes for year {year}, basin {basin}. Skipping.")
        return

    # Reproject once
    intersected_shapes = reproject_shapefile_to_equal_area(intersected_shapes)

    print(f"Processing year {year} with {len(intersected_shapes)} intersected shapes")

    for admin_shape in intersected_shapes.itertuples(index=False):
        admin_geom = admin_shape.geometry
        admin_id = admin_shape.loc_id

        # Per-admin try/except so one bad polygon doesn't kill the year.
        try:
            # split if crossing antimeridian
            geom_pieces_cea, geom_pieces_wgs84 = split_antimeridian_geom(admin_geom)
            if not geom_pieces_cea:
                geom_pieces_cea = [admin_geom]
                geom_pieces_wgs84 = [gpd.GeoSeries([admin_geom], crs="ESRI:54034").to_crs("EPSG:4326").iloc[0]]

            print(f"Split into {len(geom_pieces_cea)} pieces after antimeridian check")

            numerator_total = 0.0
            population_exposed_total = 0.0

            for piece_cea, piece_wgs84 in zip(geom_pieces_cea, geom_pieces_wgs84):
                # use precomputed WGS84 piece
                if basin != "NA":
                    piece_wgs84 = normalize_geom_to_0_360(piece_wgs84)

                paf_piece = raw_paf_raster.clip(piece_wgs84).mask(piece_wgs84, all_touched=True)
                paf_piece = subset_affected_area(paf_piece)

                # take the intersection of piece_wgs84 with the raster bounds
                minx, maxx, miny, maxy = paf_piece.bounds
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
                # Load population (bounded)
                # ----------------------------
                pop_piece = load_in_gridded_population(year, 100, bounds=intersection_cea_bounds)

                try:
                    pop_piece_masked = pop_piece.mask(piece_cea, all_touched=True)
                except rasterio.errors.WindowError:
                    print("[INFO] Admin piece does not intersect raster → skipping piece")
                    continue

                # Extract arrays
                pop_arr = pop_piece_masked._ndarray.astype(np.float32, copy=False)

                # --- resample PAF to masked grid ---
                paf_arr = (
                    paf_piece
                    .resample_to(pop_piece_masked, resampling="nearest")
                    ._ndarray.astype(np.float32, copy=False)
                )

                # ----------------------------
                # Valid cells
                # ----------------------------
                valid_mask = (pop_arr > 0) & np.isfinite(pop_arr) & (paf_arr > 0) & np.isfinite(paf_arr)

                if valid_mask.any():
                    numerator_total += (pop_arr[valid_mask] * paf_arr[valid_mask]).sum()
                    population_exposed_total += pop_arr[valid_mask].sum()

            # ----------------------------
            # Population denominator
            # ----------------------------
            pop_sum, special_region_flag = get_population_total(
                pop_df=pop_df,
                year=year,
                admin_id=admin_id,
            )

            if pop_sum == 0:
                weighted_paf = 0.0
                population_exposed = 0.0
            else:
                weighted_paf = numerator_total / pop_sum
                population_exposed = population_exposed_total

            paf_records.append({
                "location_id": admin_id,
                "year": year,
                "total_population": float(pop_sum),
                "population_exposed": float(population_exposed),
                "population_weighted_paf": float(weighted_paf),
                "relative_risk": relative_risk,
                "special_region_flag": special_region_flag,
            })
        except Exception as e:
            print(
                f"❌ Admin {admin_id} failed in basin {basin}, year {year}: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            continue

    final_paf_df = (
        pd.DataFrame.from_records(paf_records)
        .sort_values(["location_id", "year"])
        .reset_index(drop=True)
    )

    save_batch_paf_dataframe(
        paf_df=final_paf_df,
        source_id=SOURCE_ID,
        variant_label=VARIANT_LABEL,
        sample_name=sample_name,
        relative_risk=relative_risk,
        experiment_id=EXPERIMENT_ID,
        year=year,
        basin=basin,
        save_root=SAVE_ROOT,
    )



process_single_year(
    year=year,
    basin=basin,
    relative_risk=relative_risk,
)
