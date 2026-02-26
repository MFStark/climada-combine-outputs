from pathlib import Path
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
import rasterra as rt # type: ignore
import pandas as pd  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
import geopandas as gpd  # type: ignore
from shapely.geometry import Point, box, mapping  # type: ignore
from affine import Affine  # type: ignore
import os
import warnings
from collections.abc import Iterator
import argparse
import zarr # type: ignore
import dask.array as da  # type: ignore
import gc
from rra_tools.parallel import run_parallel  # type: ignore
import time

parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--storm_draw", type=str, required=True, help="Storm draw number storm_0000 to storm_0099")
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
ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage0")# TEST
SAVE_ROOT = Path("/mnt/share/scratch/users/mfiking/outputs/climada/stage1") # TEST

##########################################
#          Helper Functions              #
##########################################

def chmod_recursive(path: Path, mode: int = 0o775):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)


def iter_storms(draw_store: Path):
    for storm_path in draw_store.iterdir():
        if storm_path.is_dir() and storm_path.name.startswith("storm_"):
            yield xr.open_zarr(storm_path, consolidated=False)


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

def knots_to_ms(knots):
    """
    Convert wind speed from knots to meters per second.
    
    Parameters:
    -----------
    knots : float, array-like, or xarray.DataArray
        Wind speed in knots
        
    Returns:
    --------
    float, array-like, or xarray.DataArray
        Wind speed in meters per second
        
    Notes:
    ------
    Conversion factor: 1 knot = 0.514444 m/s
    """
    return knots * 0.514444

def interpolate_rr_from_windspeed(intensity_array, rr_samples_df, sample_name, min_windspeed_knots=25):
    """
    Interpolate relative risk values for windspeed intensity array using a specific sample.
    
    Parameters:
    -----------
    intensity_array : xarray.DataArray
        Wind intensity values in m/s
    rr_samples_df : pandas.DataFrame
        DataFrame with windspeed (knots), type, and sample columns
    sample_name : str
        Name of the sample column to use (e.g., 'sample_001')
    min_windspeed_knots : float
        Minimum windspeed threshold in knots (default: 25)
        
    Returns:
    --------
    xarray.DataArray
        Relative risk values interpolated for the intensity array
    """
    
    # Convert minimum windspeed to m/s for comparison
    min_windspeed_ms = knots_to_ms(min_windspeed_knots)
    
    # Get windspeed and RR values from the sample
    windspeed_knots = rr_samples_df['windspeed'].values
    windspeed_ms = knots_to_ms(windspeed_knots)
    rr_values = rr_samples_df[sample_name].values
    
    # Create interpolation function
    rr_interp = interp1d(
        windspeed_ms, 
        rr_values, 
        kind='linear', 
        bounds_error=False, 
        fill_value='extrapolate'
    )
    
    # Create copy to preserve coordinates and metadata
    result = intensity_array.copy()
    
    # Get min and max windspeed values from RR data
    min_rr_windspeed_ms = windspeed_ms.min()
    max_rr_windspeed_ms = windspeed_ms.max()
    max_rr_value = rr_values[np.argmax(windspeed_ms)]  # RR value at highest windspeed
    
    # Initialize all values to 0
    rr_interpolated = np.zeros_like(intensity_array.values)
    
    # Create masks for different windspeed ranges
    below_min_mask = intensity_array.values < min_rr_windspeed_ms
    above_max_mask = intensity_array.values > max_rr_windspeed_ms
    interpolation_mask = (intensity_array.values >= min_rr_windspeed_ms) & (intensity_array.values <= max_rr_windspeed_ms)
    
    # Set values below minimum to 0 (already initialized to 0)
    # rr_interpolated[below_min_mask] = 0  # Already 0
    
    # Set values above maximum to the highest RR value
    if np.any(above_max_mask):
        rr_interpolated[above_max_mask] = max_rr_value
    
    # Interpolate values within the RR data range
    if np.any(interpolation_mask):
        rr_values_interp = rr_interp(intensity_array.values[interpolation_mask])
        rr_interpolated[interpolation_mask] = rr_values_interp
    
    # Update the data array values
    result.values = rr_interpolated
    result.name = f"relative_risk_{sample_name}"
    
    return result
    
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

def storm_primary_year(storm_ds: xr.Dataset) -> int:
    """
    Return the primary year of a storm, defined as the start_date year.

    Parameters
    ----------
    storm_ds : xr.Dataset
        A single storm dataset.

    Returns
    -------
    int
        Year of storm start_date.
    """
    return pd.to_datetime(storm_ds.attrs["start_date"]).year

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
        ROOT_PATH
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



def load_relative_risk_df(relative_risk: str,root: Path = Path("/mnt/share/homes/mfiking/github_repos/climada_python/data/")):

    if relative_risk == "indirect_resp_draw":
        relative_risk_df = pd.read_csv(root / f"rd_rr_samples.csv")
    elif relative_risk == "indirect_cvd_draw":
        relative_risk_df = pd.read_csv(root / f"cvd_rr_samples.csv")
    
    return relative_risk_df


def generate_days_impact_from_intensity(
    intensity_da: xr.DataArray,
    impact_days: float = 20.0,
) -> xr.DataArray:
    """
    Create synthetic days_impact raster from intensity.

    Valid intensity pixels → impact_days
    Invalid / zero intensity → 0
    """

    data = intensity_da.values

    # Define impacted pixels
    mask = np.isfinite(data) & (data > 0)

    days = np.zeros_like(data, dtype=np.float32)
    days[mask] = impact_days

    da_days = xr.DataArray(
        days,
        coords=intensity_da.coords,
        dims=intensity_da.dims,
        name="days_impact",
        attrs=intensity_da.attrs,
    )

    da_days.attrs.update({
        "description": "Synthetic impact duration derived from intensity mask",
        "impact_days_assumed": impact_days,
        "definition": "Pixels with valid windspeed assigned fixed duration",
    })

    return da_days

##########################################
#     Subset Raster to Affected Area     #
##########################################

def subset_affected_area(
    rr_raster: rt.RasterArray,
    threshold: float = 0.0,
    buffer_pixels: int = 1,  # buffer by N raster cells
) -> rt.RasterArray:
    """
    Subset a RasterArray to the minimal bounding box
    where RR > threshold, using rasterra.clip().
    Buffers by N pixels in the raster's CRS (EPSG:4326 if geographic).
    """
    data = np.asarray(rr_raster._ndarray)

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

    # Buffer by 1 pixel in degrees
    pixel_width = abs(a)
    pixel_height = abs(e)
    pixel_buffer = max(pixel_width, pixel_height) * buffer_pixels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf["geometry"] = gdf.geometry.buffer(pixel_buffer)
        
    return rr_raster.clip(gdf)

##########################################
#        Calculate Relative Risk         #
##########################################

def generate_relative_risk(
    da_intensity: xr.DataArray,
    rr_samples_df,
    sample_name: str,
    min_windspeed_knots: float = 25.0,
) -> xr.DataArray:
    """
    Generate per-storm pixel-level relative risk from storm intensity.

    Parameters
    ----------
    storm_intensity : xr.DataArray
        Each DataArray has dims ('lat', 'lon') and values in m/s.
        Represents per-pixel maximum wind speed during the storm.
    rr_samples_df : pandas.DataFrame
        Relative risk lookup table with 'windspeed' column in knots
        and sample columns (e.g. 'sample_001').
    sample_name : str
        Column name in rr_samples_df to use.
    min_windspeed_knots : float
        Minimum windspeed threshold below which RR = 0.

    Returns
    -------
    list[xr.DataArray]
        One DataArray per storm with dims ('lat', 'lon').
    """

    storm_name = da_intensity.attrs["storm_name"]
    
    # Interpolate RR from windspeed (Katrina logic)
    da_rr = interpolate_rr_from_windspeed(
        intensity_array=da_intensity,
        rr_samples_df=rr_samples_df,
        sample_name=sample_name,
        min_windspeed_knots=min_windspeed_knots,
    )

    da_rr.attrs.update({
        "description": (
            "Pixel-level relative risk derived from storm maximum wind speed"
        ),
        "storm_name": storm_name,
        "start_date": da_intensity.attrs.get("start_date"),
        "end_date": da_intensity.attrs.get("end_date"),
        "basin": da_intensity.attrs.get("basin"),
        "category": da_intensity.attrs.get("category"),
        "rr_sample": sample_name,
        "min_windspeed_knots": min_windspeed_knots,
        "definition": (
            "Relative risk interpolated from windspeed using empirical RR curves; "
            "intensity is maximum per-pixel wind speed during storm lifetime"
        ),
    })


    return da_rr

##########################################
#          Save Yearly-Basin Raster      #
##########################################

def save_raster(
    raster_data: np.ndarray,
    template_raster: rt.RasterArray,
    storm_draw: str,
    source_id: str,
    variant_label: str,
    sample_name: str,
    relative_risk: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    year: int,
    tc_risk_draw: int,
    metric: str,  # "raw_paf" or "raw_rr"
    save_root: Path = SAVE_ROOT,
    max_retries: int = 3,
    retry_delay: float = 1.0,
):
    """
    Generic function to save raster data as GeoTIFF, with retries.

    Parameters
    ----------
    raster_data : np.ndarray
        2D array to save.
    template_raster : rt.RasterArray
        Raster template to copy CRS, transform, and no_data_value.
    metric : str
        Metric name, e.g., "raw_paf" or "raw_rr".
    tc_risk_draw : int
        Draw index (used for filename suffix).
    """
    raster_data = raster_data.astype(np.float32)

    save_dir = save_root / storm_draw / source_id / variant_label / experiment_id / batch_year / str(year) / basin / metric
    save_dir.mkdir(parents=True, exist_ok=True)

    draw_text = "" if tc_risk_draw == 0 else f"_e{tc_risk_draw - 1}"
    start_year, end_year = batch_year.split("-")
    filename = (
        f"draw_mean_{metric}_{storm_draw}_{relative_risk}_{sample_name}_{basin}_{source_id}_"
        f"{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}{draw_text}.tif"
    )
    save_path = save_dir / filename

    raster_array = rt.RasterArray(
        data=raster_data,
        transform=template_raster.transform,
        crs=template_raster.crs,
        no_data_value=template_raster.no_data_value,
    )

    # Retry loop for robust saving
    for attempt in range(max_retries):
        try:
            raster_array.to_file(
                save_path,
                driver="GTiff",
                compress="deflate",
                predictor=3,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            print(f"Saved {metric} raster as TIFF: {save_path}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Save failed for {save_path}, retrying in {retry_delay}s ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to save {save_path} after {max_retries} attempts") from e
            
##########################################
#          Main Stage 2 Function         #
##########################################



def process_single_draw(draw):
    """
    Process a single draw of storms and return yearly raw PAF and yearly RR rasters.
    """
    (
        storm_draw,
        source_id,
        variant_label,
        experiment_id,
        batch_year,
        basin,
        draw,
        relative_risk,
        sample_name,
    ) = draw
    
    intensity_draw_store = get_draw_zarr_path(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        metric="intensity",
    )
    
    rr_samples_df = load_relative_risk_df(
        relative_risk=relative_risk,
    )

    # Initialize lists to store yearly rasters
    yearly_paf = []
    yearly_rr = []

    for year in all_years_in_draw(intensity_draw_store):

        storms_in_year = [
            storm_ds
            for storm_ds in iter_storms_from_draw(intensity_draw_store)
            if year in range(
                pd.to_datetime(storm_ds.attrs["start_date"]).year,
                pd.to_datetime(storm_ds.attrs["end_date"]).year + 1,
            )
        ]

        if len(storms_in_year) == 0:
            # Append empty rasters if no storms affect this year
            # Use template from first storm of draw if needed
            first_storm = next(iter(iter_storms_from_draw(intensity_draw_store)), None)
            if first_storm is None:
                continue  # no storms at all in this draw
            template_raster = to_raster(
                ds=first_storm["intensity"],
                no_data_value=np.nan,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326",
            )
            yearly_paf.append(np.zeros_like(template_raster._ndarray))
            yearly_rr.append(np.ones_like(template_raster._ndarray))
            continue

        storms_in_year = sorted(
            storms_in_year,
            key=lambda ds: pd.to_datetime(ds.attrs["start_date"])
        )

        # Use the first storm's raster as a template for shape/resolution
        first_storm = storms_in_year[0]
        template_raster = to_raster(
            ds=first_storm["intensity"],
            no_data_value=np.nan,
            lat_col="lat",
            lon_col="lon",
            crs="EPSG:4326"
        )

        # Cumulative raster for the  for raw pafs
        one_minus_raw_paf = np.ones(
            template_raster._ndarray.shape,
            dtype=np.float32,
        )        

        # Cumulative raster for realtive risk
        rr_weighted_sum = np.zeros_like(
            template_raster._ndarray,
            dtype=np.float32,
        )
        
        # Cumulative raster for number of storms
        n_storms = np.zeros_like(
            template_raster._ndarray,
            dtype=np.int16,
        )


        for storm_ds in storms_in_year:
            storm_id = storm_ds.attrs.get("storm_id")


            # ------------------------------
            # 1. Compute relative risk
            # ------------------------------
            rr_da = generate_relative_risk(
                da_intensity=storm_ds["intensity"],
                rr_samples_df=rr_samples_df,
                sample_name=sample_name,
            )

            # ------------------------------
            # 1a. Rasterize relative risk
            # ------------------------------
            storm_rr = to_raster(
                ds=rr_da,
                no_data_value=np.nan,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326"
            )

            # ------------------------------
            # 1b. Resample to Basin Template
            # ------------------------------
            storm_rr = storm_rr.resample_to(
                target=template_raster,
                resampling="nearest",
            )

            # rr_values = relative risk raster
            rr_values = np.asarray(storm_rr._ndarray)

            # ------------------------------
            # 2. Generate days_impact raster for this storm
            # ------------------------------
            days_impact_da = generate_days_impact_from_intensity(
                storm_ds["intensity"],
                impact_days=20.0,
            )

            storm_days_impact = to_raster(
                ds=days_impact_da,
                no_data_value=0,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326"
            )

            storm_days_impact = storm_days_impact.resample_to(
                target=template_raster,
                resampling="nearest",
            )

            # days impact
            t_impact = np.asarray(storm_days_impact._ndarray)


            # Mask valid pixels
            mask = (
                np.isfinite(t_impact)
                & np.isfinite(rr_values)
                & (t_impact > 0)
                & (rr_values != 0)
            )

            if not mask.any():
                print(f"⚠️ Storm {storm_id} has no valid impacted pixels. Skipping...")
                # Clean up memory for this storm
                del storm_rr
                del days_impact_da
                del storm_days_impact
                gc.collect()
                continue  # skip this storm

            # Initialize PAF array
            paf_raw = np.zeros_like(t_impact, dtype=float)

            # Compute per-pixel PAF for this storm
            paf_raw[mask] = (rr_values[mask] - 1) / rr_values[mask] * (t_impact[mask] / 365)

            # multiply into cumulative raster
            one_minus_raw_paf[mask] *= 1 - paf_raw[mask]

            # ------------------------------
            # 5. Compute Relative Risk
            # -----------------------------
            rr_storm = np.ones_like(rr_values, dtype=np.float32)  # initialize to 1
            rr_storm[mask] = rr_values[mask]  # only affected pixels
            rr_weighted_sum = da.where(
                mask,
                rr_weighted_sum + (rr_storm - 1),
                rr_weighted_sum
            )
            n_storms = da.where(
                mask,
                n_storms + 1,
                n_storms
            )


            # Clean up
            del storm_rr
            del days_impact_da
            del storm_days_impact
            t_impact = None
            rr_values = None
            paf_raw = None
            mask = None
            rr_storm = None
            gc.collect()

        # At end of the year:
        raw_paf_year = 1 - one_minus_raw_paf

        # Initialize yearly RR raster
        rr_year = np.ones_like(rr_weighted_sum, dtype=np.float32)

        valid = n_storms > 0

        rr_year = da.where(
            valid,
            rr_weighted_sum - (n_storms - 1),
            1.0,
        )

        yearly_paf.append(raw_paf_year)
        yearly_rr.append(rr_year)

        # clean up
        one_minus_raw_paf = None
        rr_weighted_sum = None
        n_storms = None
        gc.collect()

    print(f"Completed draw {draw} for basin {basin}, batch {batch_year}")

    return yearly_paf, yearly_rr, template_raster

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
    draws = list(range(0, 99))

    # Define batch size (10 draws at a time)
    batch_size = 10

    start_year, end_year = map(int, batch_year.split("-"))
    n_years = end_year - start_year + 1

    # Initialize cumulative arrays to hold sums across all 100 draws
    cumulative_paf = [None] * n_years
    cumulative_rr = [None] * n_years
    template_raster = None

    for batch_start in range(0, len(draws), batch_size):
        batch_draws = draws[batch_start: batch_start + batch_size]

        # Prepare arguments for process_single_draw
        draw_args = [
            (
                storm_draw,
                source_id,
                variant_label,
                experiment_id,
                batch_year,
                basin,
                draw,
                relative_risk,
                sample_name,
            )
            for draw in batch_draws
        ]

        # Run parallel for this batch
        batch_results = run_parallel(
            runner=process_single_draw,
            arg_list=draw_args,
            num_cores=num_cores,
        )

        # batch_results is a list of tuples: (yearly_paf_list, yearly_rr_list, template_raster) per draw
        for draw_yearly_paf, draw_yearly_rr, draw_template in batch_results:

            if template_raster is None:
                template_raster = draw_template

            for year_idx in range(n_years):
                # Initialize cumulative arrays if first draw
                if cumulative_paf[year_idx] is None:
                    cumulative_paf[year_idx] = np.zeros_like(draw_yearly_paf[year_idx], dtype=np.float32)
                    cumulative_rr[year_idx] = np.zeros_like(draw_yearly_rr[year_idx], dtype=np.float32)

                cumulative_paf[year_idx] += draw_yearly_paf[year_idx]
                cumulative_rr[year_idx] += draw_yearly_rr[year_idx]

    # After summing all draws, take the average
    final_paf = [arr / len(draws) for arr in cumulative_paf]
    final_rr = [arr / len(draws) for arr in cumulative_rr]

    # Save to disk per year & metric
    metrics = ["raw_paf", "raw_rr"]
    for year_idx, year in enumerate(range(int(start_year), int(end_year) + 1)):
        for metric, arr in zip(metrics, [final_paf[year_idx], final_rr[year_idx]]):
            # Save raster using generic function
            save_raster(
                raster_data=arr,
                template_raster=template_raster,
                storm_draw=storm_draw,
                source_id=source_id,
                variant_label=variant_label,
                sample_name=sample_name,
                relative_risk=relative_risk,
                experiment_id=experiment_id,
                batch_year=batch_year,
                basin=basin,
                year=year,
                tc_risk_draw=0,  # or your current draw index
                metric=metric,
                save_root=save_root,
            )

            # Fix permissions
            out_path = save_root / storm_draw / source_id / variant_label / experiment_id / batch_year / str(year) / basin / metric
            chmod_recursive(out_path, mode=0o775)
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
