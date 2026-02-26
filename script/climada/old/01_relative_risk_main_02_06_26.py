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

parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
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
basin = args.basin
draw = args.draw
relative_risk = args.relative_risk
sample_name = args.sample_name

# Constants
ROOT_PATH = Path("/mnt/share/scratch/users/mfiking/climada_test_outputs_stage_0/") # TEST
SAVE_ROOT = Path("/mnt/share/scratch/users/mfiking/climada_test_outputs_stage_1/") # TEST

##########################################
#          Helper Functions              #
##########################################
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
    relative_risk_df = pd.read_csv(root / f"{relative_risk}_rr_samples.csv")

    return relative_risk_df


def load_days_impact_for_storm(
    storm_id: int,
    days_impact_draw_store: Path,
) -> xr.DataArray:
    """
    Load days_impact DataArray by matching storm_name attribute.
    """
 
    # list groups via zarr, not pathlib
    root = zarr.open(days_impact_draw_store, mode="r")

    for group_name in root.group_keys():
        ds = xr.open_zarr(
            days_impact_draw_store,
            group=group_name,
            consolidated=False,
            decode_timedelta=False,
        )
        if ds.attrs.get("storm_id") == storm_id:
            if "days_impact" not in ds:
                raise KeyError(f"'days_impact' missing in {group_name}")
            return ds["days_impact"]

    raise KeyError(f"No storm found with storm_id={storm_id}")


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
def save_raw_paf_raster_tif(
    raw_paf: np.ndarray,
    template_raster: rt.RasterArray,
    source_id: str,
    variant_label: str,
    sample_name: str,
    relative_risk: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    year: int,
    tc_risk_draw: int,
    save_root: Path = SAVE_ROOT,
):

    save_dir = save_root / source_id / variant_label / experiment_id / batch_year / str(year) / basin / "raw_paf"
    save_dir.mkdir(parents=True, exist_ok=True)

    draw_text = "" if tc_risk_draw == 0 else f"_e{tc_risk_draw - 1}"

    start_year, end_year = batch_year.split("-")
    # Build directory path
    filename = f"paf_{relative_risk}_{sample_name}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}{draw_text}.tif"

    save_path = save_dir / filename

    # Convert to RasterArray
    paf_raster = rt.RasterArray(
        data=raw_paf,
        transform=template_raster.transform,
        crs=template_raster.crs,
        no_data_value=template_raster.no_data_value,
    )

    # Save as GeoTIFF
    paf_raster.to_file(
        save_path,
        driver="GTiff",
        compress="deflate",
        predictor=3,      # good for float data
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    os.chmod(save_path, 0o775)

    print(f"Saved raw PAF raster as TIFF: {save_path}")

def save_raw_rr_raster_tif(
    raw_rr: np.ndarray,
    template_raster: rt.RasterArray,
    source_id: str,
    variant_label: str,
    sample_name: str,
    relative_risk: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    year: int,
    tc_risk_draw: int,
    save_root: Path = SAVE_ROOT,
):

    save_dir = save_root / source_id / variant_label / experiment_id / batch_year / str(year) / basin / "raw_rr"
    save_dir.mkdir(parents=True, exist_ok=True)

    draw_text = "" if tc_risk_draw == 0 else f"_e{tc_risk_draw - 1}"

    start_year, end_year = batch_year.split("-")
    # Build directory path
    filename = f"rr_{relative_risk}_{sample_name}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}{draw_text}.tif"

    save_path = save_dir / filename

    # Convert to RasterArray
    rr_raster = rt.RasterArray(
        data=raw_rr,
        transform=template_raster.transform,
        crs=template_raster.crs,
        no_data_value=template_raster.no_data_value,
    )

    # Save as GeoTIFF
    rr_raster.to_file(
        save_path,
        driver="GTiff",
        compress="deflate",
        predictor=3,      # good for float data
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    os.chmod(save_path, 0o775)

    print(f"Saved raw RR raster as TIFF: {save_path}")


##########################################
#          Main Stage 2 Function         #
##########################################

def main(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    relative_risk: str,
    sample_name: str,
):
    intensity_draw_store = get_draw_zarr_path(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        metric="intensity",
    )

    days_impact_draw_store = get_draw_zarr_path(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,

        metric="days_impact",
    )

    rr_samples_df = load_relative_risk_df(
        relative_risk=relative_risk,
    )

# ------------------------------
# Loop over all years spanned by storms
# ------------------------------
# For each year, we select storms that **affect that year**, even if they started in a previous year
# Example: A storm that starts in Dec 2005 and ends in Jan 2006:
#   - During the 2005 iteration, it is included because 2005 is within its start–end year range
#   - During the 2006 iteration, it is included again because 2006 is within its start–end year range
#
# This works because:
#   - 'intensity' raster is **storm-lifetime maximum** → can be reused across years safely
#   - 'days_impact' raster is **year-specific** → selecting `days_impact_da.sel(time=str(year))` ensures
#       that only the portion of the storm that occurs in the current year contributes to PAF
#
# As a result:
#   - Each year’s basin-level raw PAF raster correctly accumulates contributions only for days 
#     within that year, even for storms spanning multiple years

    for year in all_years_in_draw(intensity_draw_store):

        storms_in_year = [
            storm_ds
            for storm_ds in iter_storms_from_draw(intensity_draw_store)
            if year in range(
                pd.to_datetime(storm_ds.attrs["start_date"]).year,
                pd.to_datetime(storm_ds.attrs["end_date"]).year + 1,
            )
        ]

        storms_in_year = sorted(
            storms_in_year,
            key=lambda ds: pd.to_datetime(ds.attrs["start_date"])
        )
        # if no storms affect this year, skip   
        if len(storms_in_year) == 0:
            continue  # skip this year

        # Initialize a basin-level raster of ones
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
            print(f"Processing year {year}, storm {storm_id}")


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
            
            # ------------------------------
            # 2. Load days impact for this storm
            # ------------------------------
            days_impact_da = load_days_impact_for_storm(
                storm_id=storm_id,
                days_impact_draw_store=days_impact_draw_store
            )

            days_impact_year_da = days_impact_da.sel(
                time=year  # selects only the current year
            )

            # ------------------------------
            # 2a. Rasterize days impact
            # ------------------------------
            days_impact_rr = to_raster(
                ds=days_impact_year_da,
                no_data_value=0,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326"
            )

            # ------------------------------
            # 2b. Resample to Basin Template
            # ------------------------------
            storm_days_impact = days_impact_rr.resample_to(
                target=template_raster,
                resampling="nearest",
            )

            # ------------------------------
            # 4. Compute per-storm
            # ------------------------------

            # t_impact = days affected per pixel by this storm
            t_impact = np.asarray(storm_days_impact._ndarray)

            # rr_values = relative risk raster
            rr_values = np.asarray(storm_rr._ndarray)

            # Mask valid pixels
            mask = (
                np.isfinite(t_impact)
                & np.isfinite(rr_values)
                & (t_impact > 0)
                & (rr_values != 0)
            )

            # Initialize PAF array
            paf_raw = np.zeros_like(t_impact, dtype=float)

            # Compute per-pixel PAF for this storm
            paf_raw[mask] = (rr_values[mask] - 1) / rr_values[mask] * (t_impact[mask] / 365)

            # multiply into cumulative raster
            one_minus_raw_paf[mask] *= 1 - paf_raw[mask]

            # ------------------------------
            # 5. Compute Relative Risk
            # -----------------------------
            # Compute per-pixel PAF for this storm
            paf_raw[mask] = (rr_values[mask] - 1) / rr_values[mask] * (t_impact[mask] / 365)

            # multiply into cumulative raster
            one_minus_raw_paf[mask] *= 1 - paf_raw[mask]

        # At end of the year:
        raw_paf_year = 1 - one_minus_raw_paf

        # Initialize yearly RR raster
        rr_year = np.ones_like(rr_weighted_sum, dtype=np.float32)

        valid = n_storms > 0

        rr_year = da.where(
            valid,
            rr_weighted_sum - (n_storms - 1),
            0.0,
        )


        # save basin-level raw PAF raster for this year
        save_raw_paf_raster_tif(
            raw_paf=raw_paf_year,
            template_raster=template_raster,
            source_id=source_id,
            variant_label=variant_label,
            sample_name=sample_name,
            relative_risk=relative_risk,
            experiment_id=experiment_id,
            batch_year=batch_year,
            basin=basin,
            year=year,
            tc_risk_draw=draw,
        )

        save_raw_rr_raster_tif(
            raw_rr=rr_year,
            template_raster=template_raster,
            source_id=source_id,
            variant_label=variant_label,
            sample_name=sample_name,
            relative_risk=relative_risk,
            experiment_id=experiment_id,
            batch_year=batch_year,
            basin=basin,
            year=year,
            tc_risk_draw=draw,
        )



main(
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
    batch_year=batch_year,
    basin=basin,
    draw=draw,
    relative_risk=relative_risk,
    sample_name=sample_name,
)