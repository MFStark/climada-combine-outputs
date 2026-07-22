from pathlib import Path
from typing import NamedTuple
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
import rasterra as rt  # type: ignore
import pandas as pd  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
from affine import Affine  # type: ignore
import os
import argparse
import gc
import re
import time
import traceback
from rra_tools.parallel import run_parallel  # type: ignore
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
ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1_v2")
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage2_v2")

# Fixed impact duration per storm-affected pixel (days), expressed as a
# year-fraction for the PAF formula: PAF = (RR - 1) / RR * (days / 365).
# Previously materialized as a per-storm raster filled with 20.0 inside the
# storm footprint, resampled to the basin template, and indexed by mask;
# the mask reduces to `rr > 0` and the value is constant, so we just use
# the precomputed fraction directly.
IMPACT_DAYS_FRACTION = 20.0 / 365.0

class StormMeta(NamedTuple):  # type: ignore
    storm_path: Path
    start_year: int
    end_year: int
    storm_id: str

def iter_storms_metadata(draw_store: Path) -> list[StormMeta]:
    """Return a list of StormMeta for each storm in the draw, without loading full data."""
    if not draw_store.exists():
        raise FileNotFoundError(f"Draw store not found: {draw_store}")

    storm_paths = sorted(
        p for p in draw_store.iterdir() if p.is_dir() and p.name.startswith("storm_")
    )

    storms_meta = []
    for storm_path in storm_paths:
        ds = None
        try:
            ds = xr.open_zarr(storm_path, consolidated=False, chunks={})  # lazy read, no load
            start_year = pd.to_datetime(ds.attrs["start_date"]).year
            end_year = pd.to_datetime(ds.attrs["end_date"]).year
            storm_id = ds.attrs.get("storm_id", storm_path.name)
            storms_meta.append(StormMeta(storm_path, start_year, end_year, storm_id))
        finally:
            if ds is not None:
                ds.close()
    return storms_meta

def map_storms_to_years(storms_meta: list[StormMeta], years: list[int]):
    storms_by_year = {year: [] for year in years}
    for storm in storms_meta:
        for year in range(storm.start_year, storm.end_year + 1):
            if year in storms_by_year:
                storms_by_year[year].append(storm.storm_path)
    return storms_by_year

    
##########################################
#          Helper Functions              #
##########################################

def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.assign_coords(
        lon=(((ds.lon + 180) % 360) - 180)
    ).sortby("lon")

    return ds


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

class RRInterpolator(NamedTuple):
    """
    Pre-built relative-risk interpolation context, shared across all storms
    in a single draw. Built once at the top of `process_single_draw` so
    every storm in that draw reuses the same `interp1d` callable and bounds
    instead of rebuilding them.
    """
    interp: object       # interp1d callable over windspeed_ms -> RR
    min_ms: float        # lowest windspeed in the RR table (m/s)
    max_ms: float        # highest windspeed in the RR table (m/s)
    max_rr: float        # RR value at the highest windspeed (used as a cap)
    sample_name: str     # column name in rr_samples_df, kept for metadata


def build_rr_interpolator(
    rr_samples_df: pd.DataFrame,
    sample_name: str,
) -> RRInterpolator:
    """Construct the per-draw RR interpolation context."""
    windspeed_ms = knots_to_ms(rr_samples_df["windspeed"].values)
    rr_values = rr_samples_df[sample_name].values

    interp = interp1d(
        windspeed_ms,
        rr_values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    return RRInterpolator(
        interp=interp,
        min_ms=float(windspeed_ms.min()),
        max_ms=float(windspeed_ms.max()),
        max_rr=float(rr_values[np.argmax(windspeed_ms)]),
        sample_name=sample_name,
    )


def interpolate_rr_from_windspeed(
    intensity_array: xr.DataArray,
    rr_ctx: RRInterpolator,
) -> xr.DataArray:
    """
    Interpolate relative risk for a windspeed intensity array (m/s) using a
    pre-built RR interpolation context. Pixels outside the RR table's
    windspeed range get 0 (below min) or the table's max RR (above max).
    """
    intensity_values = intensity_array.values
    rr_interpolated = np.zeros_like(intensity_values)

    above_max_mask = intensity_values > rr_ctx.max_ms
    interpolation_mask = (
        (intensity_values >= rr_ctx.min_ms) & (intensity_values <= rr_ctx.max_ms)
    )

    if np.any(above_max_mask):
        rr_interpolated[above_max_mask] = rr_ctx.max_rr

    if np.any(interpolation_mask):
        rr_interpolated[interpolation_mask] = rr_ctx.interp(
            intensity_values[interpolation_mask]
        )

    # Construct a fresh DataArray with the same coords/dims/attrs as the input.
    # dict(intensity_array.attrs) makes a shallow copy so downstream attr
    # mutation (in generate_relative_risk) doesn't bleed into the source DA.
    return xr.DataArray(
        rr_interpolated,
        coords=intensity_array.coords,
        dims=intensity_array.dims,
        attrs=dict(intensity_array.attrs),
        name=f"relative_risk_{rr_ctx.sample_name}",
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

def load_relative_risk_df(relative_risk: str,root: Path = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/relative_risk_samples/")):

    if relative_risk == "indirect_resp_draw":
        relative_risk_df = pd.read_csv(root / f"rd_rr_samples.csv")
    elif relative_risk == "indirect_cvd_draw":
        relative_risk_df = pd.read_csv(root / f"cvd_rr_samples.csv")
    
    return relative_risk_df


##########################################
#        Calculate Relative Risk         #
##########################################

def generate_relative_risk(
    da_intensity: xr.DataArray,
    rr_ctx: RRInterpolator,
) -> xr.DataArray:
    """
    Generate per-storm pixel-level relative risk from storm intensity using
    the pre-built per-draw RR interpolation context.
    """
    storm_name = da_intensity.attrs.get("storm_name", "none")

    da_rr = interpolate_rr_from_windspeed(da_intensity, rr_ctx)

    da_rr.attrs.update({
        "description": "Pixel-level relative risk derived from storm maximum wind speed",
        "storm_name": storm_name,
        "start_date": da_intensity.attrs.get("start_date"),
        "end_date": da_intensity.attrs.get("end_date"),
        "basin": da_intensity.attrs.get("basin"),
        "category": da_intensity.attrs.get("category"),
        "rr_sample": rr_ctx.sample_name,
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

    """
    raster_data = raster_data.astype(np.float32)

    save_dir = save_root / storm_draw / source_id / variant_label / experiment_id / batch_year / str(year) / basin / metric
    save_dir.mkdir(parents=True, exist_ok=True)

    start_year, end_year = batch_year.split("-")
    filename = (
        f"draw_mean_{metric}_{storm_draw}_{relative_risk}_{sample_name}_{basin}_{source_id}_"
        f"{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}.tif"
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

    return save_path


def _is_valid_raster(path: Path) -> bool:
    try:
        _ = rt.load_raster(path)
        return True
    except Exception:
        return False   


def check_if_draw_complete(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    relative_risk: str,
    sample_name: str,
    save_root: Path = SAVE_ROOT,
):
    """
    Check if the expected output rasters for a draw already exist.

    Returns True if all expected rasters are present, False otherwise.
    """
    start_year, end_year = batch_year.split("-")
    years = range(int(start_year), int(end_year) + 1)

    for year in years:
        save_dir = save_root / storm_draw / source_id / variant_label / experiment_id / batch_year / str(year) / basin / "raw_paf"
        filename = (
            f"draw_mean_raw_paf_{storm_draw}_{relative_risk}_{sample_name}_{basin}_{source_id}_"
            f"{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}.tif"
        )
        save_path = save_dir / filename
        if not save_path.exists():
            return False

        if save_path.stat().st_size < 1024:
            save_path.unlink(missing_ok=True)
            print(f"⚠️ Found invalid raster (size < 1KB): {save_path}. Deleting and marking draw as incomplete.")
            return False

        if not _is_valid_raster(save_path):
            save_path.unlink(missing_ok=True)
            print(f"⚠️ Found invalid raster (failed to load): {save_path}. Deleting and marking draw as incomplete.")
            return False

    return True

def get_year_status(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    relative_risk: str,
    sample_name: str,
    save_root: Path = SAVE_ROOT,
):
    start_year, end_year = batch_year.split("-")
    years = range(int(start_year), int(end_year) + 1)

    valid_years = []
    invalid_years = []

    for year in years:
        save_dir = (
            save_root / storm_draw / source_id / variant_label /
            experiment_id / batch_year / str(year) / basin / "raw_paf"
        )

        filename = (
            f"draw_mean_raw_paf_{storm_draw}_{relative_risk}_{sample_name}_{basin}_{source_id}_"
            f"{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}.tif"
        )

        path = save_dir / filename

        if not path.exists():
            invalid_years.append(year)
            continue

        if path.stat().st_size < 1024 or not _is_valid_raster(path):
            path.unlink(missing_ok=True)
            invalid_years.append(year)
        else:
            valid_years.append(year)

    return valid_years, invalid_years
##########################################
#          Main Stage 2 Function         #
##########################################

def process_single_draw(draw):
    """
    Process a single draw of storms and return yearly raw PAF and yearly RR rasters.
    Uses storm metadata to avoid loading full datasets until necessary.
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
        template_raster,
        rr_samples_df,
        invalid_years,
    ) = draw

    # Build the RR interpolation context once per worker (per draw). Every
    # storm in this draw reuses the same interp1d + bounds instead of
    # rebuilding them per storm.
    rr_ctx = build_rr_interpolator(rr_samples_df, sample_name)

    # Path to the intensity draw
    intensity_draw_store = get_draw_zarr_path(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        metric="intensity",
    )

    # If no intensity data exists, return empty rasters
    if intensity_draw_store is None or not any(intensity_draw_store.iterdir()):
        yearly_paf = {year: np.zeros_like(template_raster._ndarray, dtype=np.float32) for year in invalid_years}
        print(f"⚠️ Draw {draw} has no intensity data. Returning empty rasters for basin {basin}, batch {batch_year}")
        return yearly_paf



    # Initialize output dictionaries
    yearly_paf = {}

    # --- STEP 1: Get metadata for all storms in the draw ---
    storms_meta = iter_storms_metadata(intensity_draw_store)  # returns list of StormMeta
    storms_by_year = map_storms_to_years(storms_meta, invalid_years)

    # --- STEP 2: Process each year individually ---
    for year in invalid_years:
        storm_paths_in_year = storms_by_year[year]

        # Skip years with no storms
        if not storm_paths_in_year:
            yearly_paf[year] = np.zeros_like(template_raster._ndarray, dtype=np.float32)
            continue

        # Initialize cumulative arrays for this year
        sum_raw_paf = np.zeros_like(template_raster._ndarray, dtype=np.float32)

        # --- STEP 3: Process each storm ---
        for storm_path in storm_paths_in_year:
            storm_id = storm_path.name  # fallback identifier if open fails
            try:
                with xr.open_zarr(storm_path, consolidated=False, chunks="auto") as storm_ds:
                    storm_ds = ensure_min_grid(storm_ds)
                    storm_id = storm_ds.attrs.get("storm_id", storm_path.name)

                    if basin == "NA":
                        storm_ds = normalize_dataset(storm_ds)

                    rr_da = generate_relative_risk(
                        da_intensity=storm_ds["intensity"],
                        rr_ctx=rr_ctx,
                    )
                    rr_da = rr_da.where(rr_da > 0)

                    storm_rr = to_raster(
                        ds=rr_da,
                        no_data_value=np.nan,
                        lat_col="lat",
                        lon_col="lon",
                        crs="EPSG:4326",
                    ).resample_to(target=template_raster, resampling="nearest")
                    rr_values = storm_rr._ndarray

                    # Pixels with a positive, finite RR contribute. NumPy `>`
                    # returns False on NaN, so `rr_values > 0` already excludes
                    # the NaN pixels from `where(rr_da > 0)` above.
                    mask = rr_values > 0
                    if mask.any():
                        # Every masked pixel sees a constant 20-day impact, so
                        # the year-fraction collapses to IMPACT_DAYS_FRACTION.
                        sum_raw_paf[mask] += (
                            (rr_values[mask] - 1) / rr_values[mask] * IMPACT_DAYS_FRACTION
                        )

                    del storm_rr, mask, rr_values, rr_da
            except Exception as e:
                print(
                    f"❌ Storm {storm_id} failed in draw {draw} "
                    f"({source_id}/{variant_label}/{experiment_id}/{batch_year}/"
                    f"{basin}, year {year}): {type(e).__name__}: {e}"
                )
                traceback.print_exc()
                continue

        # End of year calculations
        yearly_paf[year] = sum_raw_paf

        # Clean up
        del sum_raw_paf
        gc.collect()

    print(f"Completed draw {draw} for basin {basin}, batch {batch_year}")
    return yearly_paf

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
    
    # check if output already exists for this draw - if so, skip processing
    if check_if_draw_complete(
        storm_draw=storm_draw,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        relative_risk=relative_risk,
        sample_name=sample_name,
        save_root=save_root,
    ):
        print(f"✅ Output rasters already exist for draw {storm_draw}. Skipping processing.")
        return None

    valid_years, invalid_years = get_year_status(
        storm_draw=storm_draw,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        relative_risk=relative_risk,
        sample_name=sample_name,
        save_root=save_root,
    )

    if len(invalid_years) == 0:
        print(f"✅ All years valid for draw {storm_draw}. Skipping.")
        return None

    print(f"Recomputing years: {invalid_years}")
    draws = list(range(100))

    # Define batch size based on number of cores
    batch_size = num_cores

    # Generate basin-wide template raster once
    template_raster = generate_basin_template_raster(basin, res=0.1)
    # Load relative risk table
    rr_samples_df = load_relative_risk_df(relative_risk=relative_risk)

    # Initialize cumulative dictionaries to hold sums across draws
    cumulative_paf = {
        year: np.zeros_like(template_raster._ndarray, dtype=np.float32)
        for year in invalid_years
    }

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
                template_raster,
                rr_samples_df,
                invalid_years,
            )
            for draw in batch_draws
        ]

        # Run parallel for this batch
        batch_results = run_parallel(
            runner=process_single_draw,
            arg_list=draw_args,
            num_cores=num_cores,
        )
        print(f"Parallel job stage done for draw_batch: {batch_draws}")

        # batch_results is a list of yearly_paf dictionaries returned from each draw
        for draw_yearly_paf in batch_results:
            for year in invalid_years:
                arr = draw_yearly_paf.get(year)
                if arr is not None:
                    cumulative_paf[year] += arr

    # After summing all draws, take the average
    n_draws = len(draws)
    final_paf = {year: arr / n_draws for year, arr in cumulative_paf.items()}

    # Save to disk per year, collecting the written paths for a single chmod
    # sweep at the end (avoids one os.walk per year).
    saved_paths: list[Path] = []
    for year in invalid_years:
        arr = final_paf[year]

        save_path = save_raster(
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
            metric="raw_paf",
            save_root=save_root,
        )
        saved_paths.append(save_path)

    # Fix permissions on the files we wrote (and their immediate parents so
    # the group can traverse). No recursive walk needed — we know exactly
    # which paths exist.
    for path in saved_paths:
        try:
            os.chmod(path, 0o664)
            os.chmod(path.parent, 0o775)
        except Exception as e:
            print(f"⚠️ Could not set permissions for {path}: {e}")

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
