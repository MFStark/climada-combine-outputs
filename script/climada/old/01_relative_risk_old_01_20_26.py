from pathlib import Path
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
import gc
import re
import rasterra as rt # type: ignore
import datetime as dt
import pandas as pd  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
import geopandas as gpd  # type: ignore
from shapely.geometry import Point, box, mapping  # type: ignore
from affine import Affine  # type: ignore
import os
import warnings
from datetime import timedelta
from collections.abc import Iterator
import argparse

parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--draw", type=int, required=True, help="Draw index")

# Parse arguments
args = parser.parse_args()
source_id = args.source_id
variant_label = args.variant_label
experiment_id = args.experiment_id
batch_year = args.batch_year
basin = args.basin
draw = args.draw

# Constants
ROOT_PATH = Path("/mnt/share/scratch/users/mfiking/climada_storm")

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

def initialize_year_calendar_like(
    template_raster: rt.RasterArray,
    no_data_value: float = np.nan,
) -> rt.RasterArray:
    """
    Initialize a per-year calendar raster on the same grid as a template raster.
    """

    if not isinstance(template_raster, rt.RasterArray):
        raise TypeError(
            f"Expected RasterArray, got {type(template_raster)}"
        )

    # IMPORTANT: use the ndarray, not .data (memoryview)
    data = np.zeros_like(template_raster._ndarray, dtype=float)

    return rt.RasterArray(
        data=data,
        transform=template_raster.transform,
        crs=template_raster.crs,
        no_data_value=no_data_value,
    )

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


def add_to_year_calendar(year_calendar, storm_crop, days_to_add):
    a, b, c, d, e, f = year_calendar.transform[:6]
    calendar_xmin, calendar_ymax = year_calendar.x_min, year_calendar.y_max

    sc, sf = storm_crop.transform[2], storm_crop.transform[5]

    row0 = int(round((calendar_ymax - sf) / abs(e)))
    col0 = int(round((sc - calendar_xmin) / a))

    row1 = row0 + days_to_add.shape[0]
    col1 = col0 + days_to_add.shape[1]

    year_calendar._ndarray[row0:row1, col0:col1] += days_to_add


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
) -> Path:
    """
    Locate draw-level storm Zarr store produced by Stage 1.
    """
    start_year, end_year = batch_year.split("-")
    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    draw_store = (
        ROOT_PATH
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"storm_intensity_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
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


def load_shapefiles():
    shapefile=gpd.read_parquet('/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/shapes_gbd_2025.parquet')

    return shapefile


def load_relative_risk_df(relative_risk: str,root: Path = Path("/mnt/share/homes/mfiking/github_repos/climada_python/data/")):
    relative_risk_df = pd.read_csv(root / f"{relative_risk}_rr_samples.csv")

    return relative_risk_df

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

    # Buffer by 1 pixel in degrees
    pixel_width = abs(a)
    pixel_height = abs(e)
    pixel_buffer = max(pixel_width, pixel_height) * buffer_pixels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf["geometry"] = gdf.geometry.buffer(pixel_buffer)
        
    return rr_raster.clip(gdf)

##########################################
#     Intersect Shapefiles with data     #
##########################################

def intersect_shapefile_with_storm_data(shapefile_gdf, rr_sample, buffer=0):
    """
    Find shapefile rows that intersect with the relative risk data grid.

    Parameters
    ----------
    shapefile_gdf : geopandas.GeoDataFrame
        Shapefile already in target CRS
    rr_sample : xarray.DataArray
        2D raster (projected or geographic)
    buffer : float, optional
        Buffer distance in CRS units (meters if projected, degrees if geographic)

    Returns
    -------
    geopandas.GeoDataFrame
        Subset of shapefile that intersects RR data
    """

    # Identify spatial dims robustly
    x_dim, y_dim = rr_sample.rio.x_dim, rr_sample.rio.y_dim

    # Mask valid RR cells
    valid_mask = (rr_sample > 0) & rr_sample.notnull()

    # Extract coordinates
    xs, ys = np.meshgrid(
        rr_sample[x_dim].values,
        rr_sample[y_dim].values,
        indexing="xy"
    )

    valid_xs = xs[valid_mask.values]
    valid_ys = ys[valid_mask.values]

    print(f"Found {len(valid_xs)} non-zero RR grid cells")

    # Build GeoDataFrame of points in RR CRS
    data_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(valid_xs, valid_ys),
        crs=rr_sample.rio.crs,
    )

    # Optional buffer (units follow CRS!)
    if buffer > 0:
        data_gdf.geometry = data_gdf.geometry.buffer(buffer)
        print(f"Applied buffer of {buffer} CRS units")

    # CRS consistency check
    if shapefile_gdf.crs != data_gdf.crs:
        raise ValueError(
            f"CRS mismatch: shapefile={shapefile_gdf.crs}, rr={data_gdf.crs}"
        )

    # Spatial join
    intersections = gpd.sjoin(
        shapefile_gdf,
        data_gdf,
        how="inner",
        predicate="intersects",
    )

    result = shapefile_gdf.loc[intersections.index.unique()].copy()

    print(f"Found {len(result)} intersecting admin units")

    return result


##########################################
#        Compute Days Impact             #
##########################################

def compute_days_impact_for_year(
    storms_in_year: list,
    default_days: int = 20,
) -> list[np.ndarray]:
    """
    Compute days_impact per pixel for all storms in a year,
    accounting for overlapping storms.
    
    Parameters
    ----------
    storms_in_year : list of dicts
        Each dict has:
            - "storm_id"
            - "start_date" (pd.Timestamp)
            - "rr_mask" (boolean ndarray)
    default_days : int
        Default duration of each storm.
        
    Returns
    -------
    list of np.ndarray
        Each array corresponds to the storm's days_impact per pixel.
    """

    # Sort storms by start date
    storms_in_year.sort(key=lambda s: s["start_date"])
    
    days_impact_list = []
    
    for i, storm_i in enumerate(storms_in_year):
        mask_i = storm_i["rr_mask"]
        start_i = storm_i["start_date"]
        end_i = start_i + timedelta(days=default_days)
        
        # Initialize days impact array
        days_impact = np.full(mask_i.shape, default_days, dtype=np.int16)
        
        # Check future storms
        for storm_j in storms_in_year[i+1:]:
            start_j = storm_j["start_date"]
            mask_j = storm_j["rr_mask"]
            
            # If storm_j starts after storm_i ends, no truncation needed
            if start_j >= end_i:
                break
            
            # Compute pixel overlap
            overlap_pixels = mask_i & mask_j
            
            # Days until next storm starts
            delta_days = (start_j - start_i).days
            
            # Truncate days_impact for overlapping pixels
            days_impact[overlap_pixels] = np.minimum(days_impact[overlap_pixels], delta_days)
        
        days_impact_list.append(days_impact)
    
    return days_impact_list

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
    draw_store = get_draw_zarr_path(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    )

    rr_samples_df = load_relative_risk_df(
        relative_risk=relative_risk,
    )

    shapefile = load_shapefiles()

    for year in all_years_in_draw(draw_store):

        # get template storm for this year
        template_storm = next(
            (storm_ds for storm_ds in iter_storms(draw_store) if year == storm_primary_year(storm_ds)),
            None
        )
        if template_storm is None:
            continue

        # rasterize template storm relative risk
        template_raster = to_raster(
            template_storm["relative_risk"],
            no_data_value=np.nan,
            lat_col="lat",
            lon_col="lon",
            crs="EPSG:4326"
        )

        # Initialize the per-year calendar to track days impacted per pixel
        year_calendar = initialize_year_calendar_like(template_raster, year)

        # Filter storms that belong to this year
        # Filter storms for the year
        storms_in_year = [
            storm_ds for storm_ds in iter_storms(draw_store)
            if year == storm_primary_year(storm_ds)
        ]

        # Sort storms chronologically by start date
        storms_in_year = sorted(
            storms_in_year,
            key=lambda ds: pd.to_datetime(ds.attrs["start_date"])
        )

        storm_count = 0
        for storm_ds in storms_in_year:
            storm_count += 1
            storm_id = storm_ds.attrs.get("storm_name")

            # ------------------------------
            # 1. Compute relative risk
            # ------------------------------
            rr_da = generate_relative_risk(
                da_intensity=storm_ds["relative_risk"],
                rr_samples_df=rr_samples_df,
                sample_name=sample_name,
            )

            # ------------------------------
            # 2. Rasterize relative risk
            # ------------------------------
            storm_rr = to_raster(
                ds=rr_da,
                no_data_value=np.nan,
                lat_col="lat",
                lon_col="lon",
                crs="EPSG:4326"
            )

            # Ensure CRS and spatial dims for the RR data
            rr_da = rr_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
            rr_da = rr_da.rio.write_crs("EPSG:4326", inplace=False)

            # ------------------------------
            # 3. Crop to affected area
            # ------------------------------
            storm_crop = subset_affected_area(rr_raster=storm_rr)

            # ------------------------------
            # 4. Get affected admins
            # ------------------------------
            affected_admins = intersect_shapefile_with_storm_data(
                shapefile_gdf=shapefile,
                rr_sample=rr_da,
            )

            
            # ------------------------------
            # 5. Compute days impact per pixel
            # ------------------------------
            DEFAULT_DAYS = 20

            # Clip year calendar to storm crop
            a, b, c, d, e, f = storm_crop.transform[:6] 
            nrows, ncols = storm_crop.data.shape 
            xmin = c 
            xmax = c + ncols * a 
            ymax = f 
            ymin = f + nrows * e 
            storm_geom = box(xmin, ymin, xmax, ymax)    
            year_crop = year_calendar.clip(storm_geom)

            # Convert memoryviews → ndarrays ONCE
            storm_data = np.asarray(storm_crop._ndarray)
            year_data = np.asarray(year_crop._ndarray)  # last day impacted so far

            # Mask pixels affected by this storm
            affected_mask = np.isfinite(storm_data)

            # Initialize per-storm impact days
            storm_days_impact = np.zeros_like(storm_data, dtype=float)

            # First storm in the year
            if storm_count == 1:
                # All affected pixels get default 20 days
                storm_days_impact[affected_mask] = DEFAULT_DAYS
                # Update year calendar
                year_data[affected_mask] = DEFAULT_DAYS

            else:
                # For subsequent storms
                start_date = np.datetime64(storm_ds.attrs["start_date"], "D")
                s = (start_date - np.datetime64(f"{year}-01-01", "D")).astype(int) + 1
                e = s + DEFAULT_DAYS

                # Compute effective start = later of storm start or previous impact
                effective_start = np.maximum(s, year_data)

                # Compute new days contributed by this storm
                storm_days_impact[affected_mask] = np.maximum(
                    0,
                    e - effective_start[affected_mask]
                )

                # Update year calendar ONLY where this storm adds days
                year_data[storm_days_impact > 0] = e

            # Update year calendar IN PLACE
            add_to_year_calendar(year_calendar, storm_crop, storm_days_impact)

            # Store per-storm days impact raster (optional, but now correct)
            storm_days_impact = rt.RasterArray(
                data=storm_days_impact,
                transform=storm_crop.transform,
                crs=storm_crop.crs,
                no_data_value=np.nan
            )

            # ------------------------------
            # 6. Compute per-storm, per-admin PAF
            # ------------------------------
            admin_paf_list = []  # store results for this storm

            for admin in affected_admins.itertuples():
                # t_impact = days affected per pixel by this storm
                t_impact = np.asarray(storm_days_impact._ndarray)

                # rr_values = relative risk raster
                rr_values = np.asarray(storm_crop._ndarray)

                # Initialize PAF array
                paf_raw = np.zeros_like(t_impact, dtype=float)

                # Mask valid pixels
                mask = np.isfinite(t_impact) & np.isfinite(rr_values) & (t_impact > 0) & (rr_values != 0)

                # Compute per-pixel PAF for this storm
                paf_raw[mask] = (rr_values[mask] - 1) / rr_values[mask] * (t_impact[mask] / 365)

                # Store as RasterArray
                paf_raster = rt.RasterArray(
                    data=paf_raw,
                    transform=storm_days_impact.transform,
                    crs=storm_days_impact.crs,
                    no_data_value=np.nan
                )

                # ------------------------------
                # 6.2 Aggregate to this admin
                # ------------------------------

                # Clip raster to the admin geometry
                admin_crop = paf_raster.clip(admin.geometry)
                admin_data = np.asarray(admin_crop._ndarray)

                # Compute mean PAF for this admin
                mean_paf = np.nanmean(admin_data)

                # Store admin id, year, and mean PAF
                admin_paf_list.append({
                    "admin_id": getattr(admin, "admin_id", admin.Index),  # fallback to Index
                    "year": year,
                    "mean_paf": mean_paf
                })

            # Convert per-storm admin PAFs into a DataFrame
            admin_paf_df = pd.DataFrame(admin_paf_list)

            # ------------------------------
            # 6.3 Accumulate Annual PAF Across Storms
            # ------------------------------

            # admin_paf_df contains per-storm PAF for each admin
            # We want to combine them for the year

            # Group by admin and year, then compute the complement formula:
            # Annual PAF = 1 - prod(1 - paf_per_storm)
            annual_admin_paf_list = []

            for (admin_id, year), group in admin_paf_df.groupby(["admin_id", "year"]):
                # Convert mean PAFs to annual combination using 1 - prod(1 - paf)
                paf_values = group["mean_paf"].values
                annual_paf = 1 - np.prod(1 - paf_values)  # combines multiple storms

                annual_admin_paf_list.append({
                    "admin_id": admin_id,
                    "year": year,
                    "annual_paf": annual_paf
                })

            # Convert to DataFrame
            annual_admin_paf_df = pd.DataFrame(annual_admin_paf_list)
