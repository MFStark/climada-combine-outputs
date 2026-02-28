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
import re
from rra_tools.parallel import run_parallel  # type: ignore
import time
from rasterra import RasterArray  # type: ignore

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
SAVE_ROOT = Path("/mnt/share/scratch/users/mfiking/outputs/climada/stage1_2_27") # TEST

##########################################
#          Helper Functions              #
##########################################
def to_numpy(x, template):
    if x is None:
        return np.zeros_like(template._ndarray, dtype=np.float32)
    return x.compute() if hasattr(x, "compute") else x

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

def generate_basin_template_raster(basin, res=0.1, buffer_deg=0.0):
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
) -> Path | None:
    """
    Locate draw-level storm Zarr store produced by Stage 1.
    Returns None if the draw produced no storms.
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
        return None   # ← key change - return none for 0 impact processing

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
    Accepts a precomputed basin-wide template raster for cumulative arrays.
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
        template_raster,  # new argument
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
    print(f"intensity_draw: {intensity_draw_store}")
    start_year, end_year = map(int, batch_year.split("-"))
    all_years = list(range(start_year, end_year + 1)
                     )
    # If intensity data is missing, return empty rasters
    if intensity_draw_store is None or not any(iter_storms_from_draw(intensity_draw_store)):
        yearly_paf = {year: np.zeros_like(template_raster._ndarray, dtype=np.float32) for year in all_years}
        yearly_rr = {year: np.ones_like(template_raster._ndarray, dtype=np.float32) for year in all_years}
        print(f"⚠️ Draw {draw} has no intensity data. Returning empty rasters for basin {basin}, batch {batch_year}")
        return yearly_paf, yearly_rr, template_raster


    rr_samples_df = load_relative_risk_df(relative_risk=relative_risk)

    # Initialize dictionaries for yearly rasters

    yearly_paf = {}
    yearly_rr = {}

    storms = list(iter_storms_from_draw(intensity_draw_store))
    storms_by_year = {year: [] for year in all_years}

    for storm_ds in storms:
        start_year_storm = pd.to_datetime(storm_ds.attrs["start_date"]).year
        end_year_storm = pd.to_datetime(storm_ds.attrs["end_date"]).year

        for year in range(start_year_storm, end_year_storm + 1):
            if year in storms_by_year:
                storms_by_year[year].append(storm_ds)

        
    for year in all_years:
        print(f"processing year: {year}")

        storms_in_year = storms_by_year[year]

        # ✅ Skip year if no storms and return empty rasters
        if not storms_in_year:
            yearly_paf[year] = np.zeros_like(template_raster._ndarray, dtype=np.float32)
            yearly_rr[year] = np.ones_like(template_raster._ndarray, dtype=np.float32)
            continue

        # Initialize cumulative arrays ONLY when storms exist
        one_minus_raw_paf = np.ones(template_raster._ndarray.shape, dtype=np.float32)
        rr_weighted_sum = np.zeros_like(template_raster._ndarray, dtype=np.float32)
        n_storms = np.zeros_like(template_raster._ndarray, dtype=np.int16)

        storms_in_year = sorted(
            storms_in_year,
            key=lambda ds: pd.to_datetime(ds.attrs["start_date"])
        )

        for storm_ds in storms_in_year:
            storm_id = storm_ds.attrs.get("storm_id")

            # ------------------------------
            # Compute relative risk
            # ------------------------------
            rr_da = generate_relative_risk(
                da_intensity=storm_ds["intensity"],
                rr_samples_df=rr_samples_df,
                sample_name=sample_name,
            )

            storm_rr = to_raster(
                ds=rr_da,
                no_data_value=np.nan,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326"
            ).resample_to(target=template_raster, resampling="nearest")

            rr_values = np.asarray(storm_rr._ndarray)

            # ------------------------------
            # Generate days impact
            # ------------------------------
            storm_days_impact = to_raster(
                ds=generate_days_impact_from_intensity(storm_ds["intensity"], impact_days=20.0),
                no_data_value=0,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326"
            ).resample_to(target=template_raster, resampling="nearest")

            t_impact = np.asarray(storm_days_impact._ndarray)

            # Mask valid pixels
            mask = np.isfinite(t_impact) & np.isfinite(rr_values) & (t_impact > 0) & (rr_values != 0)

            if not mask.any():
                print(f"⚠️ Storm {storm_id} has no valid impacted pixels. Skipping...")
                continue

            paf_raw = np.zeros_like(t_impact, dtype=np.float32)
            paf_raw[mask] = (rr_values[mask] - 1) / rr_values[mask] * (t_impact[mask] / 365)

            one_minus_raw_paf[mask] *= 1 - paf_raw[mask]

            rr_storm = np.ones_like(rr_values, dtype=np.float32)
            rr_storm[mask] = rr_values[mask]
            rr_weighted_sum = da.where(mask, rr_weighted_sum + (rr_storm - 1), rr_weighted_sum)
            n_storms = da.where(mask, n_storms + 1, n_storms)

            # Clean up memory
            del storm_rr, storm_days_impact, paf_raw, rr_storm, t_impact, mask
            gc.collect()

        # End of year calculations
        yearly_paf[year] = 1 - one_minus_raw_paf
        yearly_rr[year] = da.where(n_storms > 0, rr_weighted_sum - (n_storms - 1), 1.0)

        # Clean up
        del one_minus_raw_paf, rr_weighted_sum, n_storms
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
    draws = list(range(0, 9))  # or 100 draws if you like

    # Define batch size (10 draws at a time)
    batch_size = 10

    start_year, end_year = map(int, batch_year.split("-"))
    n_years = end_year - start_year + 1
    all_years = list(range(start_year, end_year + 1))

    # Generate basin-wide template raster once
    template_raster = generate_basin_template_raster(basin, res=0.1)

    # Initialize cumulative dictionaries to hold sums across draws
    cumulative_paf = {year: np.zeros_like(template_raster._ndarray, dtype=np.float32) for year in all_years}
    cumulative_rr = {year: np.zeros_like(template_raster._ndarray, dtype=np.float32) for year in all_years}

    for batch_start in range(0, len(draws), batch_size):
        batch_draws = draws[batch_start: batch_start + batch_size]
        print(f"Starting batch {batch_draws}")

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
                template_raster,  # pass basin template here
            )
            for draw in batch_draws
        ]

        # Run parallel for this batch
        batch_results = run_parallel(
            runner=process_single_draw,
            arg_list=draw_args,
            num_cores=num_cores,
        )

        # batch_results is a list of tuples: (yearly_paf_dict, yearly_rr_dict, template_raster) per draw
        for draw_yearly_paf, draw_yearly_rr, _ in batch_results:

            for year in all_years:
                arr_rr = to_numpy(draw_yearly_rr.get(year), template_raster)
                arr_paf = to_numpy(draw_yearly_paf.get(year), template_raster)

                cumulative_rr[year] += arr_rr
                cumulative_paf[year] += arr_paf

    # After summing all draws, take the average
    n_draws = len(draws)
    final_paf = {year: arr / n_draws for year, arr in cumulative_paf.items()}
    final_rr = {year: arr / n_draws for year, arr in cumulative_rr.items()}

    # Save to disk per year & metric
    metrics = ["raw_paf", "raw_rr"]
    for year in all_years:
        for metric, arr in zip(metrics, [final_paf[year], final_rr[year]]):
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
                tc_risk_draw=0,  # aggregate, or optional per-draw
                metric=metric,
                save_root=save_root,
            )

            # Fix permissions
            out_path = save_root / storm_draw / source_id / variant_label / experiment_id / batch_year / str(year) / basin / metric
            chmod_recursive(out_path, mode=0o775)

    print(f"Completed all draws for basin {basin}, batch {batch_year}")

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
