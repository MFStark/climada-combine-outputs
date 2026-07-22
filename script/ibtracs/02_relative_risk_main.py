import os
import argparse
import time
import traceback
from pathlib import Path

import xarray as xr  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import rasterra as rt  # type: ignore

from scipy.interpolate import interp1d  # type: ignore
from affine import Affine  # type: ignore
from rasterra import RasterArray  # type: ignore

parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--year", type=str, required=True, help="Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--relative_risk", type=str, required=True, help="Relative risk type")

# Parse arguments
args = parser.parse_args()
year = args.year
basin = args.basin
relative_risk = args.relative_risk

# Constants
ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage1")
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage2")
ERROR_DIR = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage2_storm_errors")


SOURCE_ID = "ibtracs"
VARIANT_LABEL = "official"
EXPERIMENT_ID = "historical"
SAMPLE_NAME = "mean"
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
    

def generate_basin_template_raster(
    year: int | str,
    basin: str,
    res: float = 0.1,
):
    # read in summary extent of all storms in the basin-year
    summary_file = (
        ROOT_PATH
        / "ibtracs"
        / "official"
        / "historical"
        / str(year)
        / basin
        / "storm_extent_summary.csv"
    )

    if not summary_file.exists():
        raise FileNotFoundError(f"Storm extent summary not found: {summary_file}")
    
    extent_df = pd.read_csv(summary_file)

    # get min/max lat/lon across all storms
    lon_min = extent_df["lon_min"].min()
    lon_max = extent_df["lon_max"].max()
    lat_min = extent_df["lat_min"].min()
    lat_max = extent_df["lat_max"].max()

    # generate template raster
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
def load_relative_risk_df(relative_risk: str,root: Path = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/relative_risk_samples/")):

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

    try:
        storm_name = da_intensity.attrs["storm_name"]
    except KeyError:
        storm_name = "none"
    
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
#         Ibtracs NC Files Functions    #
##########################################
def get_ibtracs_intensity_dir(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    root: Path = ROOT_PATH,
):
    dir_path = root / source_id / variant_label / experiment_id / str(year) / basin / "intensity"
    return dir_path


def save_raster(
    raster_data: np.ndarray,
    template_raster: rt.RasterArray,
    source_id: str,
    variant_label: str,
    relative_risk: str,
    experiment_id: str,
    basin: str,
    year: int,
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

    """
    raster_data = raster_data.astype(np.float32)

    save_dir = save_root / source_id / variant_label / experiment_id / str(year) / basin / "raw_paf"
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"draw_mean_raw_paf_{relative_risk}_{basin}_{source_id}_"
        f"{experiment_id}_{variant_label}_{year}.tif"
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
            print(f"Saved raw_paf raster as TIFF: {save_path}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Save failed for {save_path}, retrying in {retry_delay}s ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to save {save_path} after {max_retries} attempts") from e

    # tight chmod on the saved file + its parent dir (was a recursive walk
    # of the entire year/basin tree, which got expensive across many saves).
    os.chmod(save_path, 0o775)
    os.chmod(save_dir, 0o775)

def _is_valid_raster(path: Path) -> bool:
    try:
        _ = rt.load_raster(path)
        return True
    except Exception:
        return False   


def check_if_draw_complete(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    year: str | int,
    basin: str,
    relative_risk: str,
    save_root: Path = SAVE_ROOT,
):
    """
    Check if the expected output rasters for a draw already exist.

    Returns True if all expected rasters are present, False otherwise.
    """

    save_dir = save_root / source_id / variant_label / experiment_id / str(year) / basin / "raw_paf"
    filename = (
        f"draw_mean_raw_paf_{relative_risk}_{basin}_{source_id}_"
        f"{experiment_id}_{variant_label}_{year}.tif"
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
##########################################
#          Main Stage 2 Function         #
##########################################

def process_ibtracs(
    year: str | int,
    basin: str,
    template_raster: rt.RasterArray,
    rr_samples_df: pd.DataFrame,
    sample_name: str = SAMPLE_NAME,
):

    # Path to the intensity draw
    intensity_dir = get_ibtracs_intensity_dir(
        source_id=SOURCE_ID,
        variant_label=VARIANT_LABEL,
        experiment_id=EXPERIMENT_ID,
        year=year,
        basin=basin,
    )

    storm_paths_in_year = (
        list(intensity_dir.glob("*.nc")) if intensity_dir.exists() else []
    )

    if not storm_paths_in_year:
        print(f"⚠️ No intensity .nc files found for basin {basin}, year {year}. Returning empty rasters.")
        return None

    # Initialize year_paf
    sum_raw_paf = np.zeros_like(template_raster._ndarray, dtype=np.float32)

    for storm_path in storm_paths_in_year:
        # Per-storm try/except so one bad storm doesn't kill the year.
        try:
            storm_ds = xr.open_dataset(storm_path, chunks="auto")
            storm_ds = ensure_min_grid(storm_ds)
            storm_id = storm_ds.attrs.get("storm_id", storm_path.name)

            # normalize dataset if basin = NA and if needed
            if basin == "NA":
                storm_ds = normalize_dataset(storm_ds)

            # Compute RR and days impact
            rr_da = generate_relative_risk(
                da_intensity=storm_ds["intensity"],
                rr_samples_df=rr_samples_df,
                sample_name=sample_name,
            )
            rr_da = rr_da.where(rr_da > 0)

            storm_rr = to_raster(
                ds=rr_da,
                no_data_value=np.nan,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326"
            ).resample_to(target=template_raster, resampling="nearest")
            rr_values = storm_rr._ndarray

            storm_days_impact = to_raster(
                ds=generate_days_impact_from_intensity(storm_ds["intensity"], impact_days=20.0),
                no_data_value=0,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326"
            ).resample_to(target=template_raster, resampling="nearest")
            t_impact = storm_days_impact._ndarray

            # Mask valid pixels
            mask = np.isfinite(t_impact) & np.isfinite(rr_values) & (t_impact > 0) & (rr_values != 0)
            if not mask.any():
                storm_ds.close()
                continue

            # Compute raw PAF
            sum_raw_paf[mask] += (rr_values[mask] - 1) / rr_values[mask] * (t_impact[mask] / 365)

            storm_ds.close()
        except Exception as e:
            print(
                f"❌ Storm {storm_path.name} failed in basin {basin}, year {year}: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            continue

    print(f"Completed calculation for basin {basin}, year {year}")
    return sum_raw_paf

def check_storm_outputs_exist(year: str | int, basin: str, root: Path = ROOT_PATH) -> bool:
    try:
        intensity_dir = get_ibtracs_intensity_dir(
            source_id=SOURCE_ID,
            variant_label=VARIANT_LABEL,
            experiment_id=EXPERIMENT_ID,
            year=year,
            basin=basin,
            root=root,
        )
        nc_files = list(intensity_dir.glob("*.nc"))
        return len(nc_files) > 0
    except Exception as e:
        print(f"Error checking storm outputs for year {year}, basin {basin}: {e}")
        return False

def main(
    year: str | int,
    basin: str,
    relative_risk: str,
    sample_name: str = SAMPLE_NAME,
    save_root: Path = SAVE_ROOT,
):
    # check if year and basin have storm outputs
    if not check_storm_outputs_exist(year, basin):
        print(f"⚠️ No storm outputs found for basin {basin}, year {year}. Skipping processing.")
        return

    # check if output already exists for this draw - if so, skip processing
    if check_if_draw_complete(
        source_id=SOURCE_ID,
        variant_label=VARIANT_LABEL,
        experiment_id=EXPERIMENT_ID,
        year=year,
        basin=basin,
        relative_risk=relative_risk,
        save_root=save_root,
    ):
        print(f"✅ Output rasters already exist. Skipping processing.")
        return    

    # Generate basin-wide template raster once
    template_raster = generate_basin_template_raster(year, basin, res=0.1)
    # Load relative risk table
    rr_samples_df = load_relative_risk_df(relative_risk=relative_risk)

    # process ibtracs data
    year_paf = process_ibtracs(
        year=year,
        basin=basin,
        template_raster=template_raster,
        rr_samples_df=rr_samples_df,
        sample_name=sample_name,
    )

    # if year paf is None, skip saving and return
    if year_paf is None:
        print(f"⚠️ Year PAF is None for basin {basin}, year {year}. Skipping saving.")
        return

    # save raster
    save_raster(
        raster_data=year_paf,
        template_raster=template_raster,
        source_id=SOURCE_ID,
        variant_label=VARIANT_LABEL,
        experiment_id=EXPERIMENT_ID,
        relative_risk=relative_risk,
        basin=basin,
        year=int(year),
        save_root=save_root,
    )

main(
    year=year,
    basin=basin,
    relative_risk=relative_risk,
    sample_name=SAMPLE_NAME,
)
