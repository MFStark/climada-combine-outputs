from climada.hazard import TCTracks, TropCyclone, Centroids
import xarray as xr  # type: ignore
import numpy as np # type: ignore
import re
from pathlib import Path
import rasterra as rt # type: ignore
import os

RR_CARDIO_MODEL = {
}

RR_RESP_MODEL = {
}

def prepare_custom_tctracks(ds_list: list[xr.Dataset]) -> TCTracks:
    """
    Convert custom xarray Datasets into a TCTracks object by filling in missing attributes and data variables.
    """
    prepared_list = []

    for ds in ds_list:
        # Fill missing attrs
        attrs_defaults = {
            "name": "CUSTOM_STORM",
            "sid": 0,
            "category": -1,
            "orig_event_flag": True,
            "data_provider": "custom",
            "id_no": 0,
            "max_sustained_wind_unit": "kn",
            "central_pressure_unit": "mb"
        }
        for k, v in attrs_defaults.items():
            if k not in ds.attrs:
                ds.attrs[k] = v

        # Fill missing data_vars
        n_time = len(ds.time)
        data_defaults = {
            "radius_max_wind": np.zeros(n_time),
            "radius_oci": np.zeros(n_time),
            "max_sustained_wind": np.zeros(n_time),
            "central_pressure": np.zeros(n_time),
            "environmental_pressure": np.zeros(n_time),
            "basin": np.array(["CUSTOM"]*n_time)
        }
        for var, val in data_defaults.items():
            if var not in ds.data_vars:
                ds[var] = xr.DataArray(val, coords={"time": ds.time}, dims=["time"])

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


def generate_intensity_per_storm(tc_tracks: TCTracks, centroids: Centroids) -> list[xr.DataArray]:
    """
    Generate per-storm intensity (max 3-sec gust wind speed) DataArrays
    for each tropical cyclone in tc_tracks.
    """

    # Build hazard with stored intensity
    haz = TropCyclone.from_tracks(tc_tracks, centroids=centroids, store_windfields=False)

    # Extract lat/lon from centroids
    lat = np.unique(centroids.coord[:, 0])
    lon = np.unique(centroids.coord[:, 1])
    lat = np.sort(lat)

    storm_list = []

    # Loop over storms
    for i, event in enumerate(tc_tracks.data):

        storm_name = event.name

        # --- Extract 1D intensity for this event ---
        intensity_1d = haz.intensity.toarray()[i, :]

        # Expected shape = (n_lat * n_lon)
        n_lat = len(lat)
        n_lon = len(lon)

        if intensity_1d.size != (n_lat * n_lon):
            continue

        # Reshape into lat/lon grid
        intensity_2d = intensity_1d.reshape(n_lat, n_lon)

        # Flip latitude for correct map orientation
        intensity_2d = np.flip(intensity_2d, axis=0)

        # Build DataArray
        da_intensity = xr.DataArray(
            intensity_2d,
            coords={"lat": np.flip(lat), "lon": lon},
            dims=["lat", "lon"],
            name=f"{storm_name}_intensity",
            attrs={
                "description": f"Storm {storm_name} intensity (max 3-sec gust wind speed)",
                "units": "m/s",
                "storm_name": storm_name,
                "category": getattr(event, "category", None),
            },
        )

        storm_list.append(da_intensity)

        # Free memory
        del intensity_1d, intensity_2d, da_intensity

    return storm_list


def generate_relative_risk(
    storm_intensities: list[xr.DataArray],
    rr_model: dict[str, np.ndarray] | callable,
) -> list[xr.DataArray]:
    """
    Convert storm intensity rasters into relative risk (RR) rasters.
    """
    
    rr_list = []

    # Build lookup function
    if isinstance(rr_model, dict):
        # piecewise linear interpolation
        winds = rr_model["winds"]
        rr_vals = rr_model["rr"]
        def rr_func(w):
            return np.interp(w, winds, rr_vals)
    else:
        rr_func = rr_model  # assume it's already a function

    # Apply risk model to each storm
    for da in storm_intensities:
        storm_name = da.name
        storm_cat = da.attrs.get("category", None)

        rr_data = rr_func(da.values)

        da_rr = xr.DataArray(
            rr_data,
            coords=da.coords,
            dims=da.dims,
            name=f"{storm_name}_RR",
            attrs={
                "description": f"Relative risk for storm {storm_name}",
                "units": "relative risk multiplier",
                "storm_name": storm_name,
                "category": storm_cat,
            },
        )

        rr_list.append(da_rr)

    return rr_list


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

def generate_rr_for_severity_from_storms(
    tc_tracks: TCTracks,
    basin: str,
    resolution: float,
) -> dict[str, dict[str, list[xr.DataArray]]]:
    """
    Generate relative risk rasters (cardio + respiratory) for each severity category.
    Returns structure:
    {
        "tropical": { "cardio": [...], "respiratory": [...] },
        "moderate": { "cardio": [...], "respiratory": [...] },
        "severe":   { "cardio": [...], "respiratory": [...] }
    }
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

    # Complete output dict
    severity_rr = {}

    # Step 3: Iterate over severity levels
    for severity_name, severity_track in severity_map.items():
        print(f"🌀 Processing {severity_name.upper()} cyclones ({len(severity_track.data)} tracks)...")

        # Skip severity group with no storms
        if len(severity_track.data) == 0:
            print(f"⚠️ No tracks found for {severity_name} — skipping.")
            continue

        # Generate per-storm intensity rasters
        storm_intensities = generate_intensity_per_storm(severity_track, centroids)

        # Generate cardio relative risk
        cardio_rr = generate_relative_risk(
            storm_intensities,
            rr_model=RR_CARDIO_MODEL,
        )

        # Generate respiratory relative risk
        resp_rr = generate_relative_risk(
            storm_intensities,
            rr_model=RR_RESP_MODEL,
        )

        # Store in nested dict
        severity_rr[severity_name] = {
            "cardio": cardio_rr,
            "respiratory": resp_rr,
        }

    return severity_rr

