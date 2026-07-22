from pathlib import Path
from dask.utils import F
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
from typing import NamedTuple

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


def ensure_min_grid(da: xr.DataArray, buffer_deg: float = 0.1) -> xr.DataArray:
    if da.lat.size == 1:
        c = float(da.lat.values[0])
        da = da.reindex(lat=[c-buffer_deg, c, c+buffer_deg], fill_value=0)

    if da.lon.size == 1:
        c = float(da.lon.values[0])
        da = da.reindex(lon=[c-buffer_deg, c, c+buffer_deg], fill_value=0)

    return da


def chmod_recursive(path: Path, mode: int = 0o775):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)

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
#          Save Yearly-Basin Raster      #
##########################################

def save_raster(
    raster_data: np.ndarray,
    template_raster: rt.RasterArray,
    source_id: str,
    variant_label: str,
    sample_name: str,
    relative_risk: str,
    experiment_id: str,
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

    save_dir = save_root / source_id / variant_label / experiment_id / str(year) / basin / metric
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"draw_mean_{metric}_{relative_risk}_{sample_name}_{basin}_{source_id}_"
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
            print(f"Saved {metric} raster as TIFF: {save_path}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Save failed for {save_path}, retrying in {retry_delay}s ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to save {save_path} after {max_retries} attempts") from e

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

    # set permissions to 775
    chmod_recursive(save_root / source_id / variant_label / experiment_id / str(year) / basin)

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

    # CHECK IF INTENSITY DATA EXISTS
    nc_files = list(intensity_dir.glob("*.nc")) if intensity_dir.exists() else []

    if not nc_files:
        print(f"⚠️ No intensity .nc files found for basin {basin}, year {year}. Returning empty rasters.")
        return None

    # Initialize year_paf
    sum_raw_paf = np.zeros_like(template_raster._ndarray, dtype=np.float32)

    # get all storm paths in intensity directory
    storm_paths_in_year = list(intensity_dir.glob("*.nc"))

    # --- STEP 3: Process each storm ---
    for storm_path in storm_paths_in_year:
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
            del storm_rr, storm_days_impact, rr_values, t_impact, mask
            storm_ds.close()
            del storm_ds
            continue

        # Compute raw PAF
        sum_raw_paf[mask] += (rr_values[mask] - 1) / rr_values[mask] * (t_impact[mask] / 365)

        # Clean up
        del storm_rr, storm_days_impact, t_impact, mask, rr_values
        storm_ds.close()
        del storm_ds

    print(f"Completed calculation for basin {basin}, year {year}")
    return sum_raw_paf

def main(
    year: str | int,
    basin: str,
    relative_risk: str,
    sample_name: str = SAMPLE_NAME,
    save_root: Path = SAVE_ROOT,
):
    
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
    template_raster = generate_basin_template_raster(basin, res=0.1)
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
