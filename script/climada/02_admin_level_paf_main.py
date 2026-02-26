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


parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
parser.add_argument("--year", type=int, required=True, help="Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--relative_risk", type=str, required=True, help="Relative risk type")
parser.add_argument("--sample_name", type=str, required=True, help="Sample name for relative risk")

# Parse arguments
args = parser.parse_args()
source_id = args.source_id
variant_label = args.variant_label
experiment_id = args.experiment_id
batch_year = args.batch_year
year = args.year
basin = args.basin
relative_risk = args.relative_risk
sample_name = args.sample_name



# Constants
PAF_ROOT = Path("/mnt/share/scratch/users/mfiking/climada_outputs_stage_1/") # TEST
SAVE_ROOT = Path("/mnt/share/scratch/users/mfiking/climada_outputs_stage_2/") # TEST
DRAWS = list(range(0, 99)) # 0-98

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
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    year: int,
    basin: str,
    draw: int,
    relative_risk: str,
    sample_name: str,
    paf_root: Path = PAF_ROOT,
):

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    filename = f"paf_{relative_risk}_{sample_name}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}{draw_text}.tif"

    raster_file = paf_root / source_id / variant_label / experiment_id / batch_year / str(year) / basin / "raw_paf" / filename

    raw_paf_raster = rt.load_raster(raster_file)

    return raw_paf_raster

##########################################
#           Load Shapefile               #
##########################################

def load_shapefiles():
    shapefile=gpd.read_parquet('/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/shapes_gbd_2025.parquet')

    return shapefile


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

    # ------------------------------
    # 1. Build RR>0 mask
    # ------------------------------
    rr_data = rr_raster._ndarray
    mask = np.isfinite(rr_data) & (rr_data > 0)

    if not mask.any():
        print("⚠️ No nonzero RR pixels found")
        return shapefile_gdf.iloc[0:0].copy()

    # ------------------------------
    # 2. Convert mask → polygons
    # ------------------------------
    shapes_gen = shapes(
        mask.astype(np.uint8),
        mask=mask,
        transform=rr_raster.transform,
    )

    geometries = [
        shape(geom) for geom, value in shapes_gen if value == 1
    ]

    rr_geom = gpd.GeoSeries(geometries, crs="EPSG:4326").union_all()

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
    batch_year: str,
    year: int,
    basin: str,
    draw: int,
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
        / batch_year
        / basin
        / year
        / "paf_df"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    start_year, end_year = batch_year.split("-")

    filename = f"paf_{relative_risk}_{sample_name}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}{draw_text}.parquet"
    save_path = save_dir / filename

    # Optional: enforce consistent column order
    preferred_cols = [
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


##########################################
#            MAIN FUNCTION               #
##########################################
# import time

def process_single_draw(args):
    (
        source_id,
        variant_label,
        experiment_id,
        batch_year,
        year,
        basin,
        draw,
        relative_risk,
        sample_name,
    ) = args

    shapefile = load_shapefiles()
    paf_records = []

    raw_paf_raster = load_raw_paf_raster(
        source_id,
        variant_label,
        experiment_id,
        batch_year,
        year,
        basin,
        draw,
        relative_risk,
        sample_name,
    ).to_crs("ESRI:54034")
    raw_paf_raster._ndarray = raw_paf_raster._ndarray.astype(np.float32, copy=False)

    intersected_shapes = intersect_shapefile_with_raster(
        shapefile,
        raw_paf_raster,
        buffer_degrees=0.1,
    ).to_crs("ESRI:54034")

    if len(intersected_shapes) == 0:
        # skip
        print(f"No intersected shapes for year {year}, basin {basin}. Skipping.")
        return


    print(f"Processing year {year} with {len(intersected_shapes)} intersected shapes")

    for idx, admin_shape in intersected_shapes.iterrows():
        # start_time = time.time()
        print(f"Processing admin unit {idx + 1} of {len(intersected_shapes)}")

        admin_geom = admin_shape.geometry
        admin_id = admin_shape["location_id"]

        # mask raw paf raster to admin shape
        masked_paf_raster = raw_paf_raster.clip(admin_geom)
        masked_paf_raster._ndarray = masked_paf_raster._ndarray.astype(np.float32, copy=False)
        bounds = admin_geom.bounds

        # get 100m gridded population data for the bounding box
        pop_raster = load_in_gridded_population(year, 100, bounds=bounds).clip(admin_geom)
        pop_raster._ndarray = pop_raster._ndarray.astype(np.float32, copy=False)

        pop_arr = pop_raster._ndarray
        pop_mask = np.isfinite(pop_arr) & (pop_arr > 0)
        pop_sum = pop_arr[pop_mask].sum()

        if pop_sum == 0:
            print(f"Admin ID: {admin_id} has zero population. Skipping.")
            del masked_paf_raster, pop_raster, pop_arr, pop_mask
            gc.collect()
            continue

        # downsample raw paf raster to 100m grid
        downsampled_paf_raster = masked_paf_raster.resample_to(pop_raster)
        downsampled_paf_raster._ndarray = downsampled_paf_raster._ndarray.astype(np.float32, copy=False)

        paf_arr = downsampled_paf_raster._ndarray

        # create mask where paf is finite and population is finite and greater than zero
        valid_mask = np.isfinite(paf_arr) & np.isfinite(pop_arr) & (pop_arr > 0)

        # calculate population weighted paf for the admin unit
        weighted_paf = (paf_arr[valid_mask] * pop_arr[valid_mask]).sum() / pop_sum

        paf_records.append({
            "location_id": admin_id,
            "year": year,
            "total_population": float(pop_sum),
            "population_weighted_paf": float(weighted_paf),
        })

        del pop_arr, paf_arr
        del masked_paf_raster, downsampled_paf_raster, pop_raster
        del valid_mask, pop_mask
        gc.collect()

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Processed admin ID {admin_id} in {elapsed_time:.2f} seconds")   

    del raw_paf_raster, intersected_shapes
    gc.collect()

    final_paf_df = (
        pd.DataFrame.from_records(paf_records)
        .sort_values(["location_id", "year"])
        .reset_index(drop=True)
    )

    save_batch_paf_dataframe(
        final_paf_df,
        source_id,
        variant_label,
        sample_name,
        relative_risk,
        experiment_id,
        batch_year,
        year,
        basin,
        draw,
    )

    del paf_records, final_paf_df
    gc.collect()


def main(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    year: int,
    basin: str,
    relative_risk: str,
    sample_name: str,
    save_root: Path = SAVE_ROOT,
):
    
    draw_args = [
        (
            source_id,
            variant_label,
            experiment_id,
            batch_year,
            year,
            basin,
            draw,
            relative_risk,
            sample_name,
        )
        for draw in DRAWS
    ]

    run_parallel(
        runner=process_single_draw,
        arg_list=draw_args,
        num_cores=8,   # tune based on memory
    )


    year = str(year)
    save_dir = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / year
        / "paf_df"
    )
    
    chmod_recursive(save_dir, mode=0o775)





main(
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
    batch_year=batch_year,
    year=year,
    basin=basin,
    relative_risk=relative_risk,
    sample_name=sample_name,
)

"""
Split batch years into single year process

On the granular level we will end up having 107,800,000 tasks

We are now bathcing by basin reducing the tasks by 250 resulting in 431,200 instead

We are requesting 8 cores to run 8 draws at a time per basin containing 250 draws

250/8 = 31.25 ~ 32 waves per year-basin

One task requires 1 hour and 75GB of memory and 8 cores

| Concurrent Tasks | Waves  | Total Runtime         |
| ---------------- | ------ | --------------------- |
| 25               | 17,248 | 17,248 hrs (718 days) |
| 50               | 8,624  | 8,624 hrs (359 days)  |
| 100              | 4,312  | 4,312 hrs (180 days)  |
| 200              | 2,156  | 2,156 hrs (90 days)   |
| 300              | 1,438  | 1,438 hrs (60 days)   |
| 500              | 863    | 863 hrs (36 days)     |
| 800              | 539    | 539 hrs (22 days)     |
| 1,000            | 432    | 432 hrs (18 days)     |
| 1,500            | 288    | 288 hrs (12 days)     |


"""