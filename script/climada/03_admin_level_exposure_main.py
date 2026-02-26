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
from rra_tools.parallel import run_parallel  # type: ignore
import gc

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
ROOT = Path("/mnt/share/scratch/users/mfiking/climada_outputs_stage_0/") # TEST
SAVE_ROOT = Path("/mnt/share/scratch/users/mfiking/climada_outputs_stage_3") # TEST

##########################################
#             Read in Data               #
##########################################
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

###########################################
#                Main                     #
###########################################
# import time
def process_single_draw(args):
    (
        source_id,
        variant_label,
        experiment_id,
        batch_year,
        basin,
        draw,
    ) = args

    shapefile = load_shapefiles()
    # start_time = time.time()

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

    exposure_df_list = []

    for year in all_years_in_draw(intensity_draw_store):
        print(f"[Draw {draw}] Processing year {year}")

        storms_in_year = [
            storm_ds
            for storm_ds in iter_storms_from_draw(intensity_draw_store)
            if year in range(
                pd.to_datetime(storm_ds.attrs["start_date"]).year,
                pd.to_datetime(storm_ds.attrs["end_date"]).year + 1,
            )
        ]

        if not storms_in_year:
            continue

        storms_in_year = sorted(
            storms_in_year,
            key=lambda ds: pd.to_datetime(ds.attrs["start_date"])
        )

        for storm_ds in storms_in_year:
            storm_id = storm_ds.attrs.get("storm_id")
            print(f"[Draw {draw}] Processing year {year}, storm {storm_id}")

            try:
                storm_exposure = get_exposure_storm_from_draw(
                    exposure_draw_store,
                    storm_id
                )
            except (FileNotFoundError, KeyError):
                print(f"⚠️ Missing exposure for storm {storm_id}, skipping")
                continue

            intensity_raster = to_raster(
                ds=storm_ds["intensity"],
                no_data_value=np.nan,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326",
            )
            intensity_raster._ndarray = intensity_raster._ndarray.astype(np.float32, copy=False)

            

            year_mask = storm_exposure["time"].dt.year == year

            exposure_numeric = (
                storm_exposure["exposure_hours"]
                .sel(time=year_mask)
                / np.timedelta64(1, "h")
            ).sum(dim="time").astype("float32").compute()

            exposure_raster = to_raster(
                ds=exposure_numeric,
                no_data_value=np.nan,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326",
            )
            exposure_raster._ndarray = exposure_raster._ndarray.astype(np.float32, copy=False)

            exposure_raster = exposure_raster.to_crs("ESRI:54034")
            intensity_raster = intensity_raster.resample_to(exposure_raster)
            intensity_raster._ndarray = intensity_raster._ndarray.astype(np.float32, copy=False)

            intersected_shapes = intersect_shapefile_with_raster(
                shapefile,
                exposure_raster,
                buffer_degrees=0.1,
            ).to_crs("ESRI:54034")

            for _, admin_shape in intersected_shapes.iterrows():
                print(f"[Draw {draw}] Processing admin {admin_shape['location_id']}")
                admin_geom = admin_shape.geometry
                admin_id = admin_shape["location_id"]

                masked_intensity = intensity_raster.clip(admin_geom)
                masked_exposure = exposure_raster.clip(admin_geom)

                max_wind_speed = np.nanmax(masked_intensity._ndarray)

                bounds = admin_geom.bounds
                pop_raster = load_in_gridded_population(year, 100, bounds=bounds)
                pop_raster._ndarray = pop_raster._ndarray.astype(np.float32, copy=False)
                pop_raster = pop_raster.clip(admin_geom)

                masked_exposure = masked_exposure.resample_to(pop_raster)
                masked_exposure._ndarray = masked_exposure._ndarray.astype(np.float32, copy=False)

                valid_mask = (
                    np.isfinite(pop_raster._ndarray)
                    & np.isfinite(masked_exposure._ndarray)
                    & (masked_exposure._ndarray > 0)
                )

                person_storm_hours = (
                    pop_raster._ndarray[valid_mask]
                    * masked_exposure._ndarray[valid_mask]
                ).sum()

                total_population = pop_raster._ndarray[valid_mask].sum()

                exposure_df_list.append({
                    "storm_id": storm_id,
                    "year": year,
                    "location_id": admin_id,
                    "person_storm_hours": float(person_storm_hours),
                    "total_population": float(total_population),
                    "max_wind_speed": float(max_wind_speed),
                })

                del masked_intensity._ndarray
                del masked_exposure._ndarray
                del pop_raster._ndarray

                del masked_intensity
                del masked_exposure
                del pop_raster
                del admin_geom
                gc.collect()


        del intensity_raster._ndarray
        del exposure_raster._ndarray
        del intensity_raster
        del exposure_raster
        del intersected_shapes

        gc.collect()

    if not exposure_df_list:
        return pd.DataFrame()

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
    # end_time = time.time()
    # print(f"Completed draw {draw} in {end_time - start_time:.2f} seconds")
    return None


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

    draw_args = [
        (
            source_id,
            variant_label,
            experiment_id,
            batch_year,
            basin,
            draw,
        )
        for draw in draws
    ]

    results = run_parallel(
        runner=process_single_draw,
        arg_list=draw_args,
        num_cores=10,
    )

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
    
"""
We are batching our non relative risks draws so 4725000/2 = 2362500
Then we batch our 250 draws by 50 batches so 2362500/50 = 47250 total tasks
We are requetsing 10 cores per task to run our draws in parallel
One task runs in 2 hours with 100GB of memory, with 10 cores

| Concurrent Tasks | Waves | Total Runtime        | Cluster RAM Required | Cluster Cores Required |
| ---------------- | ----- | -------------------- | -------------------- | ---------------------- |
| 25               | 1,890 | 157 days             | 2.5 TB               | 250 cores              |
| 50               | 945   | 79 days              | 5 TB                 | 500 cores              |
| 100              | 473   | 39 days              | 10 TB                | 1,000 cores            |
| 200              | 237   | 20 days              | 20 TB                | 2,000 cores            |
| 500              | 95    | 8 days               | 50 TB                | 5,000 cores            |
| 1,000            | 48    | 4 days               | 100 TB               | 10,000 cores           |
| 2,500            | 19    | 38 hours (~1.6 days) | 250 TB               | 25,000 cores           |
| 5,000            | 10    | 20 hours             | 500 TB               | 50,000 cores           |


"""