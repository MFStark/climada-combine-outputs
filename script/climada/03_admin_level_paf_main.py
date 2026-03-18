from pathlib import Path
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
import os
import rasterra as rt # type: ignore
import pandas as pd  # type: ignore
from rasterio.features import shapes  # type: ignore
from shapely.geometry import shape  # type: ignore
import geopandas as gpd  # type: ignore
import argparse
import gc
import rasterio  # type: ignore
from rra_tools.parallel import run_parallel  # type: ignore
from shapely.geometry import box
import shapely
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString
from shapely.ops import split, unary_union
import pyarrow.parquet as pq  # type: ignore

parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--storm_draw", type=str, required=True, help="Storm Draw")
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--relative_risk", type=str, required=True, help="Relative risk type")
parser.add_argument("--sample_name", type=str, required=True, help="Sample name for relative risk")
parser.add_argument("--num_cores", type=int, required=True, help="Number of cores to use for parallel processing")


# Parse arguments
args = parser.parse_args()
storm_draw = args.storm_draw
source_id = args.source_id
variant_label = args.variant_label
experiment_id = args.experiment_id
batch_year = args.batch_year
basin = args.basin
relative_risk = args.relative_risk
sample_name = args.sample_name
num_cores = args.num_cores


# Constants
PAF_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage2/")
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage3/")
# SAVE_ROOT = Path("/mnt/share/scratch/users/mfiking/outputs/stage3") # TEST

GDF_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0.parquet")
POP_TOTALS_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2021_all_years.parquet")
ANTIMERIDIAN = LineString([(180, -90), (180, 90)])

##############################
#     Helper Functions       #
##############################
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
    raw_paf_raster = subset_affected_area(raw_paf_raster)

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

##############################
#     Load Raw PAF Raster    #
##############################
def chmod_recursive(path: Path, mode: int = 0o775):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)

def load_raw_paf_raster(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    sample_name: str,
    relative_risk: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    year: int,
    metric: str = "raw_paf",
    paf_root: Path = PAF_ROOT,
):
    year = str(year)
    root_dir = paf_root / storm_draw / source_id / variant_label / experiment_id / batch_year / year / basin / metric
    start_year, end_year = batch_year.split("-")

    filename = (
        f"draw_mean_{metric}_{storm_draw}_{relative_risk}_{sample_name}_{basin}_{source_id}_"
        f"{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}.tif"
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



##########################################
#            Save Functions              #
##########################################
def save_batch_paf_dataframe(
    paf_df: pd.DataFrame,
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
        / storm_draw
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / year
        / "paf_df"
    )
    save_dir.mkdir(parents=True, exist_ok=True)


    start_year, end_year = batch_year.split("-")

    filename = f"paf_{storm_draw}_{relative_risk}_{sample_name}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}.parquet"
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

##########################################
#            MAIN FUNCTION               #
##########################################
# import time
def process_single_year(args):
    (
        storm_draw,
        source_id,
        variant_label,
        experiment_id,
        batch_year,
        year,
        basin,
        relative_risk,
        sample_name,
    ) = args

    # ---- skip if already complete ----
    if check_if_year_complete(
        storm_draw,
        source_id,
        variant_label,
        sample_name,
        relative_risk,
        experiment_id,
        batch_year,
        year,
        basin,
    ):
        print(f"Skipping year {year} (already complete)")
        return

    # load shapefile and population once per year
    shapefile = load_shapefiles()
    pop_df = load_population_dataframe()
    paf_records = []

    # load and clean PAF raster once
    raw_paf_raster = clean_paf_raster(
        load_raw_paf_raster(
            storm_draw=storm_draw,
            source_id=source_id,
            variant_label=variant_label,
            sample_name=sample_name,
            relative_risk=relative_risk,
            experiment_id=experiment_id,
            batch_year=batch_year,
            basin=basin,
            year=year,
        )
    )

    intersected_shapes = intersect_shapefile_with_raster(
        shapefile,
        raw_paf_raster,
        buffer_degrees=0.0,
    )

    if len(intersected_shapes) == 0:
        print(f"No intersected shapes for year {year}, basin {basin}. Skipping.")
        del raw_paf_raster, pop_df, shapefile
        gc.collect()
        return

    # Reproject once
    raw_paf_raster = reproject_raster_to_equal_area(raw_paf_raster)
    intersected_shapes = reproject_shapefile_to_equal_area(raw_paf_raster, intersected_shapes, basin)

    print(f"Processing year {year} with {len(intersected_shapes)} intersected shapes")

    for _, admin_shape in intersected_shapes.iterrows():

        admin_geom = admin_shape.geometry
        admin_id = admin_shape["loc_id"]

        # split if crossing antimeridian
        geom_pieces = split_antimeridian_geom(admin_geom)
        if not geom_pieces:
            geom_pieces = [admin_geom]

        numerator_total = 0.0
        population_exposed_total = 0.0

        for piece_geom in geom_pieces:

            # ----------------------------
            # Clip PAF raster to piece
            # ----------------------------
            paf_piece = raw_paf_raster.clip(piece_geom).mask(piece_geom, all_touched=True)

            # ----------------------------
            # Load population (bounded)
            # ----------------------------
            pop_piece = load_in_gridded_population(year, 100, bounds=piece_geom.bounds)
            pop_piece = pop_piece.mask(piece_geom, all_touched=True)

            pop_arr = pop_piece._ndarray.astype(np.float32, copy=False)
            paf_arr = paf_piece.resample_to(pop_piece)._ndarray.astype(np.float32, copy=False)
            del paf_piece, pop_piece
            gc.collect()

            # ----------------------------
            # Valid cells
            # ----------------------------
            valid_mask = (pop_arr > 0) & np.isfinite(pop_arr) & (paf_arr > 0) & np.isfinite(paf_arr)

            if valid_mask.any():
                numerator_total += (pop_arr[valid_mask] * paf_arr[valid_mask]).sum()
                population_exposed_total += pop_arr[valid_mask].sum()

            # clean arrays
            del pop_arr, paf_arr, valid_mask
            gc.collect()

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
            "storm_draw": storm_draw,
            "location_id": admin_id,
            "year": year,
            "total_population": float(pop_sum),
            "population_exposed": float(population_exposed),
            "population_weighted_paf": float(weighted_paf),
            "relative_risk": relative_risk,
            "special_region_flag": special_region_flag,
        })

        # explicitly clean per admin
        del numerator_total, population_exposed_total, geom_pieces, admin_geom
        gc.collect()

    # ----------------------------
    # Final cleanup
    # ----------------------------
    del raw_paf_raster, intersected_shapes, shapefile, pop_df
    gc.collect()

    final_paf_df = (
        pd.DataFrame.from_records(paf_records)
        .sort_values(["storm_draw", "location_id", "year"])
        .reset_index(drop=True)
    )

    save_batch_paf_dataframe(
        final_paf_df,
        storm_draw,
        source_id,
        variant_label,
        sample_name,
        relative_risk,
        experiment_id,
        batch_year,
        year,
        basin,
    )

    del paf_records, final_paf_df
    gc.collect()

def main(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    relative_risk: str,
    sample_name: str,
    num_cores: int,
    save_root: Path = SAVE_ROOT,
):
    start_year, end_year = batch_year.split("-")
    start_year = int(start_year)
    end_year = int(end_year)

    years = list(range(start_year, end_year + 1))
    year_args = [
        (
            storm_draw,
            source_id,
            variant_label,
            experiment_id,
            batch_year,
            year,
            basin,
            relative_risk,
            sample_name,
        )
        for year in years
    ]

    run_parallel(
        runner=process_single_year,
        arg_list=year_args,
        num_cores=num_cores,   # tune based on memory
        progress_bar=True,
    )
    print(f"parallel tasks done")



    # apply permissions for all processed years
    for year in years:
        save_dir = (
            save_root
            / storm_draw
            / source_id
            / variant_label
            / experiment_id
            / batch_year
            / basin
            / str(year)
            / "paf_df"
        )

        if save_dir.exists():
            chmod_recursive(save_dir, mode=0o775)



main(
    storm_draw=storm_draw,
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
    batch_year=batch_year,
    basin=basin,
    relative_risk=relative_risk,
    sample_name=sample_name,
    num_cores=num_cores,
)
