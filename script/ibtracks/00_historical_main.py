from pathlib import Path
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
from climada.hazard import TCTracks, TropCyclone, Centroids
import re
import rasterra as rt # type: ignore
import pandas as pd  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
import geopandas as gpd  # type: ignore
from shapely.geometry import Point, box, mapping  # type: ignore
from affine import Affine  # type: ignore
import warnings
import argparse
import gc

parser = argparse.ArgumentParser(description="Run IBTracks historical storm data processing code")

# Define arguments
parser.add_argument("--year", type=int, required=True, help="Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")

# Parse arguments
args = parser.parse_args()
year = args.year
basin = args.basin

############################################
#              Helper Functions            #
############################################
    
def normalize_lon(lon: float) -> float:
    """Normalize longitude to [-180, 180] range."""
    lon = ((lon + 180) % 360) - 180
    return lon


######################################
#    Basin Centroid Functions        #
######################################

def generate_basin_centroids(
    basin: str,
    res: float = 0.1,
    buffer_deg: float = 5.0,
) -> Centroids:
    """
    Generate Centroids for a specific tropical cyclone basin.

    - Uses 0–360 longitude convention (IBTrACS-consistent)
    - Adds a configurable buffer to avoid edge clipping
    - Safely handles storms crossing the 180° meridian
    """

    basin_bounds = {
        'EP': ['180E', '0N', '290E', '60N'],
        'NA': ['260E', '0N', '360E', '60N'],
        'NI': ['30E',  '0N', '100E', '50N'],
        'SI': ['20E',  '45S', '100E', '0S'],
        'AU': ['100E', '45S', '180E', '0S'],
        'SP': ['180E', '45S', '250E', '0S'],
        'WP': ['100E', '0N', '180E', '60N'],
    }

    if basin not in basin_bounds:
        raise ValueError(
            f"Basin '{basin}' not recognized. "
            f"Available basins: {list(basin_bounds.keys())}"
        )

    def parse_coord(coord_str: str) -> float:
        """
        Convert coordinate string (e.g. '250E', '45S') to float degrees.

        Longitude stays in 0–360 space.
        Latitude stays in [-90, 90].
        """
        match = re.match(r"([0-9\.]+)([ENWS])", coord_str)
        if not match:
            raise ValueError(f"Invalid coordinate string: {coord_str}")

        val, hemi = match.groups()
        val = float(val)

        if hemi == 'S':
            val = -val
        elif hemi == 'W':
            val = 360.0 - val  # explicit 0–360 handling

        return val

    lon_min, lat_min, lon_max, lat_max = [
        parse_coord(c) for c in basin_bounds[basin]
    ]

    # Apply buffer
    lon_min -= buffer_deg
    lon_max += buffer_deg
    lat_min -= buffer_deg
    lat_max += buffer_deg

    # Expand upper bounds to include last grid cell
    lon_max += res
    lat_max += res

    # Create centroids
    centroids = Centroids.from_pnt_bounds(
        (lon_min, lat_min, lon_max, lat_max),
        res=res,
    )

    return centroids


######################################
#    Hazard Generation Functions     #
######################################

def generate_hazard_per_track(tc_tracks: TCTracks, centroids: Centroids) -> TropCyclone:
    """
    Generate CLIMADA TropCyclone hazard object from TCTracks and Centroids.
    """

    haz = TropCyclone.from_tracks(tc_tracks, centroids=centroids, store_windfields=True)

    return haz


######################################
#    Wind Speed Generation Functions #
######################################

def generate_speed_per_storm(haz: TropCyclone, centroids: Centroids, tc_tracks: TCTracks) -> list[xr.DataArray]:
    """
    Generate per-storm wind speed DataArrays for a list of tropical cyclones.
    """

    lat = np.unique(centroids.coord[:, 0])
    lon = np.unique(centroids.coord[:, 1])
    lat = np.sort(lat)

    storm_list_speed = []

    for i, event in enumerate(tc_tracks.data):
        storm_name = event.name  # <-- Grab the storm name directly
        storm_id = event.sid
        storm_basin = event.basin
        storm_category = event.category
        times = event.time  # array of timesteps
        wf = haz.windfields[i].toarray()  # shape: (time, n_centroids, 2)

        n_time = len(times)
        n_lat = len(lat)
        n_lon = len(lon)

        try:
            wf_reshaped = wf.reshape(n_time, n_lat, n_lon, 2)
        except ValueError:
            print(f"⚠️ Skipping storm {storm_name} due to shape mismatch")
            continue

        # Preserve timestep if it exists
        timestep = getattr(event, "time_step", None)
        coords = {"time": times, "lat": np.flip(lat), "lon": lon, "dir": ["u", "v"]}
        if timestep is not None:
            coords["time_step"] = ("time", np.array(timestep))


        da = xr.DataArray(
            wf_reshaped,
            coords=coords,
            dims=["time", "lat", "lon", "dir"],
            name=f"{storm_name}_windfields"
        )

        # Compute wind speed
        da_speed = np.sqrt(da.isel(dir=0)**2 + da.isel(dir=1)**2)
        da_speed.name = storm_name  # <-- Name the DataArray after the storm
        da_speed.attrs.update({
            "description": f"Storm {storm_name} wind speed",
            "units": "m/s",
            "storm_name": storm_name,
            "storm_id": storm_id,
            "basin": storm_basin,
            "category": storm_category,
        })

        storm_list_speed.append(da_speed)

        # Free memory
        del wf, wf_reshaped, da, da_speed

    return storm_list_speed


######################################
#    Per Storm Intensity Functions   #
######################################

def generate_intensity_per_storm(
    haz: TropCyclone,
    centroids: Centroids,
    tc_tracks: TCTracks,
) -> list[xr.DataArray]:
    """
    Generate per-storm, per-pixel intensity using CLIMADA haz.intensity.

    Intensity is defined as the maximum wind speed experienced at each pixel
    during the storm lifetime.

    Returns
    -------
    list[xr.DataArray]
        One DataArray per storm with dims ('lat', 'lon')
    """
    lon = np.unique(centroids.coord[:, 1])

    lat = np.unique(centroids.coord[:, 0])
    lat = np.sort(lat)
    lat_desc = lat[::-1]   # descending, north → south

    n_lat = len(lat)
    n_lon = len(lon)

    storm_list_intensity = []

    for i, event in enumerate(tc_tracks.data):

        storm_name = event.name
        storm_basin = event.basin
        storm_category = event.category
        storm_id = event.sid


        try:
            # shape: (n_centroids,)
            intensity_flat = haz.intensity.toarray()[i, :]
        except Exception as e:
            print(f"⚠️ Could not read intensity for {storm_name}: {e}")
            continue

        if intensity_flat.size != n_lat * n_lon:
            print(
                f"⚠️ Skipping {storm_name}: grid mismatch "
                f"{intensity_flat.size} vs {n_lat * n_lon}"
            )
            continue

        # reshape to 2D grid
        intensity_2d = intensity_flat.reshape(n_lat, n_lon)

        da = xr.DataArray(
            intensity_2d,
            coords={"lat": lat_desc, "lon": lon},
            dims=["lat", "lon"],
            name=f"{storm_name}_intensity",
)


        da.attrs.update({
            "description": "Per-storm pixel-level maximum wind speed",
            "units": "m/s",
            "storm_name": storm_name,
            "storm_id": storm_id,
            "basin": storm_basin,
            "category": storm_category,
            "definition": (
                "Maximum wind speed experienced at each pixel "
                "during the storm lifetime"
            ),
        })

        storm_list_intensity.append(da)

    return storm_list_intensity


######################################
#    Per Storm Exposure Functions   #
######################################

def compute_exposure_per_storm_total(
    storm_list: list[xr.DataArray],
    wind_threshold: float = 17.0,
) -> list[xr.DataArray]:
    """
    Compute total exposure raster (in hours) per storm.
    Returns 1 DataArray per storm with:
        dims: ('lat', 'lon')
        representing total exposure over the storm lifetime.
    """
    exposure_list = []

    for storm_da in storm_list:
        storm_name = storm_da.name
        storm_basin = storm_da.basin
        storm_category = storm_da.category
        storm_id = storm_da.storm_id

        # --- Safety checks ---
        if "time_step" not in storm_da.coords:
            raise ValueError(f"'time_step' coordinate not found in storm {storm_da.name}")
        if "time" not in storm_da.coords:
            raise ValueError(f"'time' coordinate not found in storm {storm_da.name}")

        # --- Step 1: Mask where wind >= threshold ---
        mask = xr.where(storm_da > wind_threshold, 1.0, 0.0)

        # --- Step 2: Exposure per timestep (hours) ---
        exposure_raw = mask * storm_da["time_step"]

        # --- Step 3: Sum over full storm duration ---
        exposure_total = exposure_raw.sum(dim="time")

        # --- Step 4: Metadata ---
        exposure_total.attrs.update(storm_da.attrs)
        exposure_total.attrs.update({
            "description": (
                f"Total exposed hours per pixel over storm lifetime "
                f"where wind > {wind_threshold} m/s"
            ),
            "units": "hours",
            "aggregation": "storm_total",
            "wind_threshold_m_s": wind_threshold,
            "exposure_definition": (
                "Sum of timestep exposure durations per pixel over entire storm"
            ),
            "storm_name": storm_name,
            "storm_id": storm_id,
            "basin": storm_basin,
            "category": storm_category,
        })

        exposure_total.name = "exposure"

        exposure_list.append(exposure_total)

    return exposure_list


####################################
# Single Storm Exposure Functions  #
####################################

def compute_exposure(
    storm_tracks: xr.DataArray,
    wind_threshold: float = 17.0,
) -> xr.DataArray:
    """
    Compute exposure for a single storm.
    """
    exposure = compute_exposure_per_storm_total([storm_tracks], wind_threshold=wind_threshold)[0]
    return exposure


##########################################
#     Intersect Shapefiles with data     #
##########################################

def load_shapefiles():
    shapefile=gpd.read_parquet('/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/shapes_gbd_2025.parquet')

    return shapefile

def intersect_shapefile_with_rr_data(shapefile_gdf, rr_sample, buffer_degrees=0):
    """
    Find shapefile rows that intersect with the relative risk data grid.
    
    Parameters:
    -----------
    shapefile_gdf : geopandas.GeoDataFrame
        The shapefile data (already loaded)
    rr_sample : xarray.DataArray  
        2D relative risk data with dimensions [lat, lon] from single sample
    buffer_degrees : float, optional
        Buffer around data points in degrees (default: 0)
    
    Returns:
    --------
    geopandas.GeoDataFrame
        Subset of shapefile that intersects with RR data
    """

    # drop time dimension
    # rr_sample = rr_sample.isel(time=0)

    # Create points from the RR data grid where we have non-zero data
    valid_mask = (rr_sample > 0) & ~rr_sample.isnull()
    
    # Get lat/lon coordinates of valid grid cells
    lats, lons = np.meshgrid(rr_sample.lat.values, rr_sample.lon.values, indexing='ij')
    valid_lats = lats[valid_mask.values]
    valid_lons = lons[valid_mask.values]
    
    print(f"Found {len(valid_lats)} non-zero data points")
    
    # Create GeoDataFrame from valid data points
    data_points = [Point(lon, lat) for lon, lat in zip(valid_lons, valid_lats)]
    data_gdf = gpd.GeoDataFrame({'geometry': data_points}, crs='EPSG:4326')
    
    # Apply buffer if requested
    if buffer_degrees > 0:
        data_gdf.geometry = data_gdf.geometry.buffer(buffer_degrees)
        print(f"Applied {buffer_degrees}° buffer to data points")
    
    # Ensure both datasets have the same CRS
    if shapefile_gdf.crs != data_gdf.crs:
        print(f"Reprojecting shapefile from {shapefile_gdf.crs} to {data_gdf.crs}")
        shapefile_gdf = shapefile_gdf.to_crs(data_gdf.crs)
    
    # Find intersections using spatial join (faster than manual iteration)
    print("Finding intersections...")
    intersections = gpd.sjoin(shapefile_gdf, data_gdf, how='inner', predicate='intersects')
    
    # Get unique shapefile indices that intersect
    intersecting_indices = intersections.index.unique()
    result = shapefile_gdf.loc[intersecting_indices].copy()
    
    print(f"Found {len(result)} shapefile features that intersect with RR data")
    
    return result


#####################################
#         Raster Functions          #
#####################################


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

#######################################
#      Read in Gridded Population     #
#######################################


def load_in_gridded_population(year: int | str, meters: int | str, bounds: tuple | None = None):
    if bounds is None:
        pop_raster = rt.load_raster(f"/mnt/team/rapidresponse/pub/population-model/results/current/world_cylindrical_{meters}/{year}q1.tif")
        return pop_raster
    else:
        pop_raster = rt.load_raster(f"/mnt/team/rapidresponse/pub/population-model/results/current/world_cylindrical_{meters}/{year}q1.tif", 
                                bounds=bounds)
    
    return pop_raster

################################
#         Main Function        #
################################
def main(
    year: int | str,
    basin: str
):
    tracks = TCTracks.from_ibtracs_netcdf(provider="official", basin=basin, year_range=(year, year))
    year = str(year)


    if not tracks.data:
        print(f"No storms found for {basin} in {year}")
        return


    # get centroids for basin
    centroids = generate_basin_centroids(basin=basin)

    # generate hazard object
    haz = generate_hazard_per_track(tracks, centroids)

    # per storm wind speed
    storm_list_speed = generate_speed_per_storm(haz, centroids, tracks)

    # generate per-storm intensity
    storm_list_intensity = generate_intensity_per_storm(haz, centroids, tracks)

    # get admin shapefile
    shapefile_gdf = load_shapefiles()

    # Build lookup dict once
    intensity_by_id = {
        ds.storm_id: ds
        for ds in storm_list_intensity
    }

    meta_df_list = []

    for storm in storm_list_speed:
        storm_id = storm.storm_id
        year = storm_id[:4]
        start_date = storm.time.values[0]
        end_date = storm.time.values[-1]
        # format dates for printing yyyy-mm-dd
        start_date = np.datetime_as_string(start_date, unit='D')
        end_date = np.datetime_as_string(end_date, unit='D')
        print(f"Processing storm {storm_id}")

        # get associated storm intensity dataset
        intensity_ds = intensity_by_id.get(storm_id)
        if intensity_ds is None:
            print(f"⚠️  No intensity data for storm {storm_id}")
            continue

        # compute exposure
        exposure = compute_exposure(storm)

        # get intersecting admin units
        intersecting_admin_units = intersect_shapefile_with_rr_data(shapefile_gdf, exposure)
        if intersecting_admin_units.empty:
            print(f"⚠️  No intersecting admin units for storm {storm_id}")
            continue
        intersecting_admin_units_54034 = intersecting_admin_units.to_crs("ESRI:54034")

        # if intersecting admin units are found, rasterize the exposure and intensity
        exposure_raster = to_raster(
            ds=exposure,
            no_data_value=np.nan,
            lat_col="lat",
            lon_col="lon",
            crs="EPSG:4326",
        )
        try:
            cropped_exposure_raster = subset_affected_area(exposure_raster, threshold=0.0, buffer_pixels=2)
        except ValueError:
            print(f"⚠️  No affected pixels for storm {storm_id}, skipping...")
            continue

        intensity_raster = to_raster(
            ds=intensity_ds,
            no_data_value=np.nan,
            lat_col="lat",
            lon_col="lon",
            crs="EPSG:4326",
        )
        try:
            cropped_intensity_raster = subset_affected_area(intensity_raster, threshold=0.0, buffer_pixels=2)
        except ValueError:
            print(f"⚠️  No affected pixels for storm {storm_id}, skipping...")
            continue

        for location_id in intersecting_admin_units_54034['location_id'].unique():

            # use bounds from admin units to get population raster
            admin_units_loc = intersecting_admin_units_54034[
                intersecting_admin_units_54034['location_id'] == location_id
            ]
            admin_bounds = admin_units_loc.total_bounds


            # read population with bounds
            pop_raster = load_in_gridded_population(year=int(year), meters=100, bounds=admin_bounds)
            pop_vals_total = pop_raster._ndarray
            pop_total = np.nansum(pop_vals_total) # total population in admin unit

            # resample exposure and intensity to population raster 100m
            location_exposure_raster = cropped_exposure_raster.resample_to(pop_raster)
            location_intensity_raster = cropped_intensity_raster.resample_to(pop_raster)

            # subset to affected area, skip location if no affected pixels
            try:
                location_exposure_raster = subset_affected_area(location_exposure_raster, threshold=0.0, buffer_pixels=1)
                location_intensity_raster = subset_affected_area(location_intensity_raster, threshold=0.0, buffer_pixels=1)
            except ValueError:
                print(f"⚠️  No affected pixels for location {location_id}, skipping...")
                continue


            # Get max wind speed per location
            max_wind_speed = np.nanmax(np.asarray(location_intensity_raster))

            # calculate people
            pop_raster = pop_raster.resample_to(location_exposure_raster)
            mask = np.isfinite(np.asarray(location_exposure_raster)) & (np.asarray(location_exposure_raster) > 0)
            exposed_population = np.nansum(np.asarray(location_exposure_raster)[mask] * np.asarray(pop_raster)[mask])
            
            meta_df_list.append({
                "storm_id": storm_id,
                "year": year,
                "start_date": start_date,
                "end_date": end_date,
                "location_id": location_id,
                "max_wind_speed": max_wind_speed,
                "pop_total": pop_total,
                "exposed_population": exposed_population
            })
            
            # clean up in memory per location
            del location_exposure_raster
            del location_intensity_raster
            del pop_raster
        
        # clean up in memory per storm
        del exposure
        del exposure_raster
        del intensity_raster
        del cropped_exposure_raster
        del cropped_intensity_raster
        del intersecting_admin_units
        gc.collect()

    # save meta_df
    if meta_df_list:
        meta_df = pd.DataFrame(meta_df_list)
        save_root = Path("/mnt/team/rapidresponse/pub/tropical-storms/data") / "ibtracks"
        full_path = save_root / year / f"ibtracks_{basin}_historical.csv"
        full_path.parent.mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(full_path, index=False)
        full_path.chmod(0o775)
        print(f"Saved historical data to {full_path}")
    else:
        print(f"No intersecting data found for any storms in basin: {basin}, year: {year}")
        
main(
    year=year,
    basin=basin
)