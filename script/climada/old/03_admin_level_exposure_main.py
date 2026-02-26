from pathlib import Path
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

warnings.simplefilter("ignore", FutureWarning)

parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")


# Parse arguments
args = parser.parse_args()
source_id = args.source_id
variant_label = args.variant_label
experiment_id = args.experiment_id
batch_year = args.batch_year
basin = args.basin



# Constants
ROOT = Path("/mnt/share/scratch/users/mfiking/climada_test_outputs_stage_0/") # TEST
SAVE_ROOT = Path("/mnt/share/scratch/users/mfiking/climada_test_outputs_stage_3") # TEST

##########################################
#             Read in Data               #
##########################################

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
        decode_times=True,
    )

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

###########################################
#            Helper Functions             #
###########################################
def all_years_in_draw(draw_store: Path) -> list[int]:
    """
    Get all unique years covered by storms in a draw.

    Parameters
    ----------
    draw_store : Path
        Path to the draw-level Zarr store.

    Returns
    -------
    list[int]
        Sorted list of years present in the storms.
    """
    years = set()
    for storm_ds in iter_storms(draw_store):
        storm_start = pd.to_datetime(storm_ds.attrs["start_date"])
        storm_end = pd.to_datetime(storm_ds.attrs["end_date"])
        storm_years = range(storm_start.year, storm_end.year + 1)
        years.update(storm_years)
    return sorted(years)

def iter_storms(draw_store: Path):
    for storm_path in draw_store.iterdir():
        if storm_path.is_dir() and storm_path.name.startswith("storm_"):
            yield xr.open_zarr(storm_path, consolidated=False)


def iter_storms_from_draw(draw_store: Path) -> Iterator[xr.Dataset]:
    """
    Lazily iterate over storm_* Zarr groups in a draw.

    Each yielded Dataset represents a single storm.
    """
    if not draw_store.exists():
        raise FileNotFoundError(f"Draw store not found: {draw_store}")

    storm_paths = sorted(
        p for p in draw_store.iterdir()
        if p.is_dir() and p.name.startswith("storm_")
    )

    for storm_path in storm_paths:
        yield xr.open_zarr(
            storm_path,
            consolidated=False,
            chunks="auto",   # critical for raster ops
        )

###########################################
#           Raster Functions              #
###########################################
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

###########################################
#             Save Functions              #
###########################################
def save_batch_dataframe(
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
    os.chmod(save_path, 0o775)

    print(f"Saved batch-year exposure dataframe: {save_path}")

###########################################
#                Main                     #
###########################################
def main(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
):

    # load shapefiles
    shapefile = load_shapefiles()

    # path for intensity store
    intensity_draw_store = get_draw_zarr_path(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        metric="intensity",
    )

    # path for exposure store
    exposure_draw_store = get_draw_zarr_path(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,

        metric="exposure_hours",
    )

    exposure_df_list = []
    # iterate through all years in the intensity draw store
    for year in all_years_in_draw(intensity_draw_store):

        # get all storms that affect this year
        storm_intensity_in_year = [
            storm_ds
            for storm_ds in iter_storms_from_draw(intensity_draw_store)
            if year in range(
                pd.to_datetime(storm_ds.attrs["start_date"]).year,
                pd.to_datetime(storm_ds.attrs["end_date"]).year + 1,
            )
        ]

        # sort storms by start date
        storm_intensity_in_year = sorted(
            storm_intensity_in_year,
            key=lambda ds: pd.to_datetime(ds.attrs["start_date"])
        )
        # if no storms affect this year, skip   
        if len(storm_intensity_in_year) == 0:
            continue  # skip this year

        for storm_ds in storm_intensity_in_year:
            storm_id = storm_ds.attrs.get("storm_id")
            print(f"Processing year {year}, storm {storm_id}")

            # get corresponding exposure datasets for these storms
            storm_exposure_in_year = get_exposure_storm_from_draw(
                exposure_draw_store, storm_id
            )

            # rasterize storm intensity and exposure data
            intensity_raster = to_raster(
                ds=storm_ds["intensity"],
                no_data_value=np.nan,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326",
            )

            # Select the year
            year_mask = storm_exposure_in_year["time"].dt.year == year

            # Convert to numeric hours BEFORE sum
            exposure_raster_da = (
                storm_exposure_in_year["exposure_hours"]
                .sel(time=year_mask)
                / np.timedelta64(1, "h")  # convert to hours first
            )

            # Now sum over time (numeric)
            exposure_raster_numeric = exposure_raster_da.sum(dim="time").astype("float32").compute()

            # Pass numeric array to raster
            exposure_raster = to_raster(
                ds=exposure_raster_numeric,
                no_data_value=np.nan,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326",
            )


        # reproject rasters to ESRI:54034
        exposure_raster = exposure_raster.to_crs("ESRI:54034")
        # resample intensity raster to exposure raster
        intensity_raster = intensity_raster.resample_to(exposure_raster)

        # intersect both rasters with admin shapefile
        intersected_shapes = intersect_shapefile_with_raster(
            shapefile,
            exposure_raster,
            buffer_degrees=0.1,
        )

        # convert to ESRI:54034
        intersected_shapes = intersected_shapes.to_crs("ESRI:54034")

        # process by admin unit
        for idx, admin_shape in intersected_shapes.iterrows():
            admin_geom = admin_shape.geometry
            admin_id = admin_shape['location_id']  # Replace with actual admin ID column name

            # Mask intensity and exposure rasters to admin shape
            masked_intensity_raster = intensity_raster.clip(admin_geom)
            masked_exposure_raster = exposure_raster.clip(admin_geom)

            # get max windspeed from intensity data
            max_wind_speed = np.nanmax(masked_intensity_raster._ndarray)

            # load in population raster for the bounding box of the admin shape
            bounds = admin_geom.bounds  # (minx, miny, maxx, maxy)

            # load 100m gridded population data for the bounding box
            pop_raster = load_in_gridded_population(year, 100, bounds=bounds)

            # clip population raster to admin shape
            pop_raster = pop_raster.clip(admin_geom)

            # resample exposure raster to population raster
            masked_exposure_raster = masked_exposure_raster.resample_to(pop_raster)


            # create mask where population raster is finite and exposure raster is finite and greater than zero
            valid_mask = (np.isfinite(pop_raster._ndarray) & 
                            np.isfinite(masked_exposure_raster._ndarray) & 
                            (masked_exposure_raster._ndarray > 0))

            # multiply population raster with exposure raster using the mask to get person storm hours for the admin unit
            person_storm_hours = (pop_raster._ndarray[valid_mask] * masked_exposure_raster._ndarray[valid_mask]).sum()

            # get total population for the admin unit
            total_population = pop_raster._ndarray[valid_mask].sum()

            # create dataframe - storm_id, year, location_id, person_storm_hours, total_population, max_wind_speed, storm_category
            storm_admin_df = pd.DataFrame({
                'storm_id': [storm_id],
                'year': [year],
                'location_id': [admin_id],
                'person_storm_hours': [person_storm_hours],
                'total_population': [total_population],
                'max_wind_speed': [max_wind_speed],
            })

            exposure_df_list.append(storm_admin_df)

    # concatenate all dataframes in exposure_df_list
    final_exposure_df = pd.concat(exposure_df_list, ignore_index=True)
    final_exposure_df = final_exposure_df.sort_values(by=['location_id', 'year', 'storm_id']).reset_index(drop=True)

    # Save final exposure dataframe
    save_batch_dataframe(
        exposure_df=final_exposure_df,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    )
    
main(
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
    batch_year=batch_year,
    basin=basin,
)
    
