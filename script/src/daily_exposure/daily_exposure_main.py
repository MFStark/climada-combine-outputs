from climada.hazard import TCTracks, TropCyclone, Centroids
import xarray as xr  # type: ignore
import numpy as np # type: ignore
import re
from pathlib import Path
import rasterra as rt # type: ignore
import os


def prepare_minimal_tctracks(ds_list: list[xr.Dataset]) -> TCTracks:
    """
    Convert custom xarray Datasets into a TCTracks object compatible with
    TropCyclone.from_tracks for windfield/intensity calculations.

    Ensures the minimal required coordinates, data variables, and attributes.
    
    Required for TropCyclone.from_tracks:
      - Coordinates: time (lat/lon only if store_windfields=True)
      - Data variables: max_sustained_wind, central_pressure, basin
      - Attributes: name, category, sid, orig_event_flag, data_provider, id_no
    Optional variables filled with default values if missing:
      - radius_max_wind, radius_oci
    """

    prepared_list = []

    for ds in ds_list:
        # Ensure coordinates
        if "time" not in ds.coords:
            raise ValueError("Dataset must contain 'time' coordinate.")
        n_time = len(ds.time)

        # Minimal required data variables
        required_vars = {
            "max_sustained_wind": np.zeros(n_time),
            "central_pressure": np.zeros(n_time),
            "basin": np.array(["CUSTOM"] * n_time)
        }

        # Optional variables
        optional_vars = {
            "radius_max_wind": np.zeros(n_time),
            "radius_oci": np.zeros(n_time)
        }

        # Fill missing variables
        for var, val in {**required_vars, **optional_vars}.items():
            if var not in ds.data_vars:
                ds[var] = xr.DataArray(val, coords={"time": ds.time}, dims=["time"])

        # Required attributes
        attrs_defaults = {
            "name": "CUSTOM_STORM",
            "category": -1,
            "sid": 0,
            "orig_event_flag": True,
            "data_provider": "custom",
            "id_no": 0,
            "max_sustained_wind_unit": "kn",
            "central_pressure_unit": "mb"
        }
        for k, v in attrs_defaults.items():
            if k not in ds.attrs:
                ds.attrs[k] = v

        # Check lat/lon if windfields are stored
        if "lat" not in ds or "lon" not in ds:
            print("⚠️ 'lat' and 'lon' coordinates missing. Required only if store_windfields=True.")

        prepared_list.append(ds)

    return TCTracks(data=prepared_list)


def normalize_lon(lon):
    """Normalize longitude to [-180, 180] range."""
    lon = ((lon + 180) % 360) - 180
    return lon

def generate_basin_centroids(basin: str, res: float = 0.1) -> "Centroids":
    """
    Generate Centroids for a specific tropical cyclone basin.
    """

    # Dictionary of basin bounds
    basin_bounds = {
        'EP': ['180E', '0N', '290E', '60N'],
        'NA': ['260E', '0N', '360E', '60N'],
        'NI': ['30E', '0N', '100E', '50N'],
        'SI': ['20E', '45S', '100E', '0S'],
        'AU': ['100E', '45S', '180E', '0S'],
        'SA': ['180E', '45S', '250E', '0S'], # Original SA - possible mismatch
        'WP': ['100E', '0N', '180E', '60N'],
        'GL': ['0E', '90S', '360E', '90N']
    }

    if basin not in basin_bounds:
        raise ValueError(f"Basin '{basin}' not recognized. Available: {list(basin_bounds.keys())}")

    def parse_coord(coord_str):
        """Convert coordinate string with direction to float degrees."""
        match = re.match(r"([0-9\.]+)([ENWS])", coord_str)
        if not match:
            raise ValueError(f"Invalid coordinate string: {coord_str}")
        val, dir_ = match.groups()
        val = float(val)
        if dir_ in ['W', 'S']:
            val = -val
        return val

    lon_min, lat_min, lon_max, lat_max = [parse_coord(c) for c in basin_bounds[basin]]

    # Normalize longitudes to [-180, 180]
    lon_min = normalize_lon(lon_min)
    lon_max = normalize_lon(lon_max)

    # Expand upper bounds by resolution to include last grid cell
    lon_max += res
    lat_max += res

    # Create Centroids for the basin
    centroids = Centroids.from_pnt_bounds((lon_min, lat_min, lon_max, lat_max), res=res)

    return centroids


def generate_hazard_per_storm(tc_tracks: TCTracks, centroids: Centroids) -> list[xr.DataArray]:
    """
    Generate per-storm wind speed DataArrays for a list of tropical cyclones.
    """

    haz = TropCyclone.from_tracks(tc_tracks, centroids=centroids, store_windfields=True)

    lat = np.unique(centroids.coord[:, 0])
    lon = np.unique(centroids.coord[:, 1])
    lat = np.sort(lat)

    storm_list = []

    for i, event in enumerate(tc_tracks.data):
        storm_name = event.name  # <-- Grab the storm name directly
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
            "category": getattr(event, "category", None)
        })

        storm_list.append(da_speed)

        # Free memory
        del wf, wf_reshaped, da, da_speed

    return storm_list

def split_tracks_by_category(tc_tracks: TCTracks) -> dict[str, TCTracks]:
    """
    Split a TCTracks object into multiple TCTracks subsets based on the 'category' attribute.
    """
    category_severity_map = {
        "severe": [4, 5],
        "moderate": [1, 2, 3],
        "tropical": [0],
    }

    results = {}

    # Extract track list
    tracks = tc_tracks.data

    for label, cats in category_severity_map.items():
        # Select only those tracks with matching category
        filtered = [tr for tr in tracks if getattr(tr, "category", None) in cats]

        if not filtered:
            print(f"⚠️ No events found for '{label}' categories {cats}")
            continue

        # Create a new TCTracks object containing only these storms
        new_tc = TCTracks()
        new_tc.data = filtered
        results[label] = new_tc

    return results

def generate_severity_tracks(tc_tracks: TCTracks) -> tuple[TCTracks | None, TCTracks | None, TCTracks | None]:
    """
    Generate separate TCTracks objects for each severity category.
    """
    split_tracks = split_tracks_by_category(tc_tracks)
    tropical_tracks = split_tracks.get("tropical")
    moderate_tracks = split_tracks.get("moderate")
    severe_tracks = split_tracks.get("severe")

    return tropical_tracks, moderate_tracks, severe_tracks

def compute_daily_exposure_from_storm_list(
    storm_list: list[xr.DataArray],
    wind_threshold: float = 18.0,
) -> dict[str, xr.DataArray]:
    """
    Compute daily exposure rasters (hours) from a list of per-storm wind speed DataArrays.
    Uses the per-timestep duration from each storm's 'time_step' data variable.
    Mimics the reference code by using Xarray broadcasting and resample.
    """

    daily_exposures = {}

    for storm_da in storm_list:
        if "time_step" not in storm_da.coords:
            raise ValueError(f"'time_step' coordinate not found in storm {storm_da.name}")

        # --- Compute mask: 1 if speed >= threshold, else 0 ---
        mask = xr.where(storm_da >= wind_threshold, 1.0, 0.0)

        # --- Multiply mask by timestep duration ---
        exposure = mask * storm_da["time_step"]

        # --- Resample by day and sum to get daily exposure ---
        daily = exposure.resample(time="D").sum(dim="time")

        # --- Add each day to output dict ---
        for day in daily.time.values:
            day_str = str(np.datetime64(day, "D"))
            daily_exposures[day_str] = daily.sel(time=day)

    return daily_exposures


def generate_exposure_for_severity_from_storms(
    tc_tracks: TCTracks,
    basin: str,
    resolution: float,
) -> dict[str, dict[str, xr.DataArray]]:
    """
    Generate daily exposure rasters from per-storm wind speed datasets for each severity category.
    """

    # Step 1: Generate centroids for the basin
    centroids = generate_basin_centroids(basin=basin, res=resolution)

    # Step 2: Generate severity-specific tracks
    tropical_tracks, moderate_tracks, severe_tracks = generate_severity_tracks(tc_tracks)
    severity_map = {
        "tropical": tropical_tracks,
        "moderate": moderate_tracks,
        "severe": severe_tracks,
    }

    severity_daily_exposures = {}

    # Step 3: Iterate over severity levels
    for severity_name, severity_track in severity_map.items():
        print(f"🌀 Processing {severity_name.upper()} cyclones ({len(severity_track.data)} tracks)...")

        # Skip if no tracks
        if len(severity_track.data) == 0:
            print(f"⚠️ No tracks found for {severity_name} — skipping.")
            continue

        # Step 3.2: Generate per-storm wind speed datasets (list of xr.DataArrays)
        storm_list = generate_hazard_per_storm(severity_track, centroids)

        # Step 3.3: Compute daily exposure rasters using the storm list
        daily_exposures = compute_daily_exposure_from_storm_list(
            storm_list,
        )

        severity_daily_exposures[severity_name] = daily_exposures

    return severity_daily_exposures

def save_daily_exposure_rasters(severity_daily_exposures: dict, output_dir: Path):
    """
    Save daily exposure rasters for each severity level to GeoTIFF files using rioxarray
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for severity, day_dict in severity_daily_exposures.items():
        severity_dir = output_dir / severity
        severity_dir.mkdir(exist_ok=True)

        for day_str, da in day_dict.items():
            da = da.astype(np.float32)

            # Skip empty rasters
            if da.sum().item() == 0:
                continue

            # Assign proper CRS if missing
            if da.rio.crs is None:
                da = da.rio.write_crs("EPSG:4326", inplace=True)

            # Ensure proper names for rioxarray (lat->y, lon->x)
            da_rio = da.rename({"lat": "y", "lon": "x"})

            # Save GeoTIFF
            save_name = f"{day_str}_{severity}"
            out_path = severity_dir / f"{save_name}.tif"
            da_rio.rio.to_raster(out_path)

            # Optional: set file permissions
            os.chmod(out_path, 0o775)
            print(f"💾 Saved: {out_path}")

def generate_and_save_daily_exposure_rasters(
    tc_tracks: TCTracks,
    basin: str,
    resolution: float,
    output_dir: str,
):
    """
    Generate and save daily exposure rasters for each severity category.
    """

    # Generate daily exposure rasters
    severity_daily_exposures = generate_exposure_for_severity_from_storms(
        tc_tracks,
        basin,
        resolution,
    )

    # Save rasters to disk
    save_daily_exposure_rasters(severity_daily_exposures, output_dir)
