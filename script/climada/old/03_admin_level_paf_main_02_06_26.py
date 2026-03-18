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


parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
parser.add_argument("--year", type=int, required=True, help="Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--draw", type=int, required=True, help="Draw index")
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
draw = args.draw
relative_risk = args.relative_risk
sample_name = args.sample_name



# Constants
PAF_ROOT = Path("/mnt/share/scratch/users/mfiking/climada_test_outputs_stage_1/") # TEST
SAVE_ROOT = Path("/mnt/share/scratch/users/mfiking/climada_test_outputs_stage_2a/") # TEST


##############################
#     Load Raw PAF Raster    #
##############################

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
    os.chmod(save_path, 0o775)

    print(f"Saved batch-year PAF dataframe: {save_path}")


##########################################
#            MAIN FUNCTION               #
##########################################

def main(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    year: int,
    basin: str,
    draw: int,
    relative_risk: str,
    sample_name: str,
):
    shapefile = load_shapefiles()

    paf_records = []
    # Step 1: Read in year basin specific raw paf raster
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
    )

    # reproject to ESRI:54034
    raw_paf_raster = raw_paf_raster.to_crs("ESRI:54034")

    # Step 2: Intersect shapefile with raw paf raster
    intersected_shapes = intersect_shapefile_with_raster(
        shapefile,
        raw_paf_raster,
        buffer_degrees=0.1,
    )

    print(f"Processing year {year} with {len(intersected_shapes)} intersected shapes")
    # convert to ESRI:54034
    intersected_shapes = intersected_shapes.to_crs("ESRI:54034")

    # Step 3: Process by admin unit
    for idx, admin_shape in intersected_shapes.iterrows():
        print(f"Processing admin unit {idx + 1} of {len(intersected_shapes)}")
        admin_geom = admin_shape.geometry
        admin_id = admin_shape['location_id']  # Replace with actual admin ID column name

        # Mask raw paf raster to admin shape
        masked_paf_raster = raw_paf_raster.clip(admin_geom)

        # Get bounding box of admin shape
        bounds = admin_geom.bounds  # (minx, miny, maxx, maxy)

        # Load 100m gridded population data for the bounding box
        pop_raster = load_in_gridded_population(year, 100, bounds=bounds)

        # clip to admin shape
        pop_raster = pop_raster.clip(admin_geom)

        pop_arr = pop_raster._ndarray
        paf_arr = masked_paf_raster._ndarray

        # get pop sum of non NaN and >0 values
        pop_mask = (np.isfinite(pop_raster._ndarray) & (pop_raster._ndarray > 0))
        pop_sum = pop_raster._ndarray[pop_mask].sum()

        # Downsample raw paf raster to 100m grid
        downsampled_paf_raster = masked_paf_raster.resample_to(pop_raster)

        # create arrays
        pop_arr = pop_raster._ndarray
        paf_arr = downsampled_paf_raster._ndarray

        # create mask where paf is finite and population is finite and greater than zero
        valid_mask = np.isfinite(paf_arr) & (np.isfinite(pop_arr)) & (pop_arr > 0)


        # check if total population is zero
        if pop_sum == 0:
            print(f"Admin ID: {admin_id} has zero population. Skipping.")

            # 🧹 cleanup before continue
            del masked_paf_raster
            del pop_raster
            del pop_arr
            del paf_arr
            gc.collect()
            continue

        # calculate population weighted paf for the admin unit
        weighted_paf = (downsampled_paf_raster._ndarray[valid_mask] * pop_raster._ndarray[valid_mask]).sum() / pop_sum

        # create dataframe - location_id, year, total_population, population_weighted_paf
        paf_records.append({
            'location_id': admin_id,
            'year': year,
            'total_population': float(pop_sum),
            'population_weighted_paf': float(weighted_paf),
        })
        print(f"Calculated population-weighted PAF for admin ID {admin_id}")

        # ------------------------------
        # 🧹 Explicit cleanup after each admin unit
        # ------------------------------
        del masked_paf_raster
        del downsampled_paf_raster
        del pop_raster
        del pop_arr
        del paf_arr
        admin_geom = None
        pop_mask = None
        valid_mask = None
        pop_arr = None
        paf_arr = None
        downsampled_paf_raster = None
        gc.collect()
        print(f"Cleaned up memory after admin ID {admin_id}")

    # ------------------------------
    # 🧹 Explicit cleanup after each year
    # ------------------------------
    del raw_paf_raster
    del intersected_shapes
    gc.collect()
    print(f"Cleaned up memory after year {year}")



    # concatenate all dataframes in paf_df_list
    final_paf_df = pd.DataFrame.from_records(paf_records)
    final_paf_df = final_paf_df.sort_values(by=['location_id', 'year']).reset_index(drop=True)

    # Save final paf dataframe
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



main(
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
    batch_year=batch_year,
    year=year,
    basin=basin,
    draw=draw,
    relative_risk=relative_risk,
    sample_name=sample_name,
)
