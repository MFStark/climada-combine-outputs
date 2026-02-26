from asyncio.log import logger
from pathlib import Path
from venv import logger
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
from climada.hazard import TCTracks, TropCyclone, Centroids
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
import zarr  # type: ignore
import argparse
import stat
from datetime import datetime, timedelta
from numcodecs import Blosc  # type: ignore
from typing import Optional, Union, Tuple
import shutil
from rra_tools.parallel import run_parallel  # type: ignore
import gc
import concurrent.futures

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
# ROOT_PATH = Path("/mnt/share/scratch/users/mfiking/tc_risk/") # TEST
ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")
SAVE_ROOT = Path("/mnt/share/scratch/users/mfiking/climada_outputs_stage_0/") # TEST
LOG_DIR = Path("/mnt/share/scratch/users/mfiking/climada_outputs_log")
RESOLUTION = 0.1  # degrees

######################################
#        Read in Tracks              #
######################################

def read_custom_tracks(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
) -> Path:
    """
    Locate TC risk Zarr draw for CLIMADA processing.

    Returns
    -------
    Path
        Path to the draw-level Zarr store.
    """

    start_year, end_year = batch_year.split("-")

    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    track_store = (
        ROOT_PATH
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"tracks_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
    )

    if not track_store.exists():
        raise FileNotFoundError(f"Zarr store not found: {track_store}")

    return track_store



def read_storm_groups_from_draw(draw_store: Path) -> list[xr.Dataset]:
    """
    Discover and open all storm_* Zarr groups in a draw.

    Parameters
    ----------
    draw_store : Path
        Path to draw-level Zarr store.

    Returns
    -------
    List[xr.Dataset]
        List of storm-level xarray Datasets (raw, unmodified).
    """
    if not draw_store.exists():
        raise FileNotFoundError(f"Draw store not found: {draw_store}")

    storm_paths = sorted(
        p for p in draw_store.iterdir()
        if p.is_dir() and p.name.startswith("storm_")
    )

    storms = []
    for storm_path in storm_paths:
        ds_storm = xr.open_zarr(storm_path, consolidated=False)
        storms.append(ds_storm)

    return storms


def read_single_storm_from_draw(draw_store: Path, storm_index: int) -> xr.Dataset:
    """
    Open a single storm_* Zarr group from a draw.

    Parameters
    ----------
    draw_store : Path
        Path to the draw-level Zarr store.
    storm_index : int
        Storm index (0..N), corresponds to 'storm_0000', 'storm_0001', etc.

    Returns
    -------
    xr.Dataset
        Single storm dataset (raw, unmodified).
    """
    if not draw_store.exists():
        raise FileNotFoundError(f"Draw store not found: {draw_store}")

    # Build storm folder name
    storm_name = f"storm_{storm_index:04d}"
    storm_path = draw_store / storm_name

    if not storm_path.exists():
        raise FileNotFoundError(f"Storm Zarr not found: {storm_path}")

    # Open single storm dataset lazily
    ds_storm = xr.open_zarr(storm_path, consolidated=False)

    return ds_storm


def normalize_storm_for_climada(ds_storm: xr.Dataset) -> xr.Dataset:
    """
    Normalize a raw Zarr storm dataset into the exact CLIMADA-compatible
    storm Dataset expected by the existing pipeline.
    """

    # --- time coordinate ---
    time_dt = ds_storm.coords["time"].values
    n_time = time_dt.size

    # --- core variables ---
    lon = ds_storm["lon"].values
    lat = ds_storm["lat"].values
    vmax = ds_storm["max_sustained_wind"].values
    cp = ds_storm["central_pressure"].values
    env = ds_storm["environmental_pressure"].values
    basin = ds_storm["basin"].values

    # Safety: normalize longitude if needed
    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180

    # --- time step (already provided, hourly) ---
    dt_hours = float(ds_storm["time_step"].values[0])

    # --- attributes ---
    start_date_iso = (
        time_dt[0].astype("datetime64[D]").item().isoformat()
    )
    end_date_iso = (
        time_dt[-1].astype("datetime64[D]").item().isoformat()
    )

    storm_name = ds_storm.attrs.get("name", "unnamed_storm")
    storm_basin = basin[0]

    sid = int(ds_storm.attrs.get("sid", -1))
    id_no = int(ds_storm.attrs.get("id_no", sid))
    category = int(ds_storm.attrs.get("category", 0))

    # --- construct final dataset ---
    ds = xr.Dataset(
        coords={"time": time_dt},
        data_vars={
            "lon": (("time",), lon),
            "lat": (("time",), lat),
            "max_sustained_wind": (("time",), vmax),
            "central_pressure": (("time",), cp),
            "environmental_pressure": (("time",), env),
            "basin": (("time",), basin),
            # kept for backward compatibility
            "radius_max_wind": (("time",), np.zeros(n_time)),
            "radius_oci": (("time",), np.zeros(n_time)),
            "time_step": (("time",), np.full(n_time, dt_hours)),
        },
        attrs={
            "name": storm_name,
            "start_date": start_date_iso,
            "end_date": end_date_iso,
            "storm_basin": storm_basin,
            "sid": sid,
            "id_no": id_no,
            "category": category,
            "orig_event_flag": True,
            "data_provider": "custom",
            "max_sustained_wind_unit": "kn",
            "central_pressure_unit": "mb",
        },
    )

    return ds

def storm_start_from_dataset(ds: xr.Dataset) -> tuple:
    """
    Primary key: start_date
    Secondary key: sid (stable tie-breaker)
    """
    start = ds.attrs.get("start_date")
    if start is None:
        # absolute fallback
        start = ds.time.values[0]

    return (
        datetime.fromisoformat(str(start)),
        ds.attrs.get("sid", -1),
    )

def get_ordered_storm_indices(track_store: Path) -> list[int]:
    """
    Discover storms in a draw-level tracks Zarr store and return
    storm indices ordered chronologically using storm_start logic.
    """
    root = zarr.open_group(track_store, mode="r")

    storm_entries = []

    for group_name in root.group_keys():
        if not group_name.startswith("storm_"):
            continue

        try:
            storm_index = int(group_name.split("_")[1])
        except ValueError:
            continue

        # Open storm lazily via xarray (attrs only + time coord)
        ds = xr.open_zarr(track_store / group_name, consolidated=False)

        key = storm_start_from_dataset(ds)

        storm_entries.append(
            (storm_index, key)
        )

        # Explicitly close to avoid open file handles
        ds.close()

    if not storm_entries:
        raise RuntimeError(f"No storm groups found in {track_store}")

    # Sort by (start_date, sid)
    storm_entries_sorted = sorted(storm_entries, key=lambda x: x[1])

    # Extract ordered indices
    return [idx for idx, _ in storm_entries_sorted]



def prepare_zarr_files(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    storm_index: int | str = None,
) -> "TCTracks":

    # --- 1. Locate the draw-level Zarr store ---
    draw_store = read_custom_tracks(
        source_id,
        variant_label,
        experiment_id,
        batch_year,
        basin,
        draw,
    )

    # --- 2. Read storms ---
    if storm_index is not None:
        raw_storm = read_single_storm_from_draw(draw_store, storm_index)
        storm = normalize_storm_for_climada(raw_storm)
        return TCTracks(data=[storm])

    raw_storms: list[xr.Dataset] = read_storm_groups_from_draw(draw_store)

    # --- 3. Normalize first ---
    storms = [normalize_storm_for_climada(ds) for ds in raw_storms]

    # --- 4. Sort normalized storms chronologically ---
    def storm_start(ds: xr.Dataset) -> tuple:
        """
        Primary key: start_date
        Secondary key: sid (stable tie-breaker)
        """
        start = ds.attrs.get("start_date")
        if start is None:
            # absolute fallback (should never trigger now)
            start = ds.time.values[0]

        return (
            datetime.fromisoformat(str(start)),
            ds.attrs.get("sid", -1),
        )

    storms_sorted = sorted(storms, key=storm_start)


    # --- 5. Build TCTracks ---
    return TCTracks(data=storms_sorted)


############################################
#              Helper Functions            #
############################################
    
def normalize_lon(lon: float) -> float:
    """Normalize longitude to [-180, 180] range."""
    lon = ((lon + 180) % 360) - 180
    return lon

def chmod_recursive(path: Path, mode: int = 0o775):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)



#####################################
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
        storm_start_date = event.start_date
        storm_end_date = event.end_date
        storm_basin = event.storm_basin
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
            "start_date": storm_start_date,
            "end_date": storm_end_date,
            "basin": storm_basin,
            "category": storm_category,
        })

        storm_list_speed.append(da_speed)

        # Free memory
        del wf, wf_reshaped, da, da_speed

    return storm_list_speed




######################################
#    Yearly Exposure Functions       #
######################################

def compute_yearly_exposure_per_storm(
    storm_list_speed: list[xr.DataArray],
    wind_threshold: float = 17.0,
) -> list[xr.DataArray]:
    """
    Compute per-storm, per-year exposure hours at the pixel level.

    Exposure is defined as the number of timesteps where wind speed
    is >= wind_threshold. Each timestep is assumed to represent 1 hour.

    Parameters
    ----------
    storm_list_speed : list[xr.DataArray]
        List of storm wind speed DataArrays with dims ('time', 'lat', 'lon').

    wind_threshold : float
        Wind speed threshold in m/s (default = 17 m/s).

    Returns
    -------
    list[xr.DataArray]
        One DataArray per storm with:
        - dims: ('time', 'lat', 'lon'), where time = year
        - values: exposure hours
    """

    exposure_per_storm = []

    # ============================================================
    # LOOP THROUGH STORMS (independently)
    # ============================================================
    for storm_da in storm_list_speed:

        if "time" not in storm_da.coords:
            raise ValueError(f"Storm {storm_da.name} missing 'time' coordinate")

        # --------------------------------------------------------
        # 1. Threshold → exposure mask (1 hour per timestep)
        # --------------------------------------------------------
        exposure = xr.where(storm_da >= wind_threshold, 1.0, 0.0)

        # --------------------------------------------------------
        # 2. Group by year
        # --------------------------------------------------------
        time_index = pd.DatetimeIndex(storm_da["time"].values)

        year_groups = time_index.to_period("Y").to_timestamp()

        group_da = xr.DataArray(
            year_groups,
            dims="time",
            coords={"time": exposure.time},
            name="year",
        )

        yearly_exposure = exposure.groupby(group_da).sum(dim="time")

        # normalize dimension name
        if "year" in yearly_exposure.dims:
            yearly_exposure = yearly_exposure.rename({"year": "time"})

        yearly_exposure = yearly_exposure.assign_coords(
            time=np.array(yearly_exposure.time.values, dtype="datetime64[ns]")
        )

        yearly_exposure = yearly_exposure.astype("float32")

        # --------------------------------------------------------
        # 3. Metadata (explicit + preserved)
        # --------------------------------------------------------
        yearly_exposure.name = "exposure_hours"

        yearly_exposure.attrs.update({
            # Storm identity
            "storm_name": storm_da.attrs.get("storm_name"),
            "storm_id": storm_da.attrs.get("storm_id"),
            "start_date": storm_da.attrs.get("start_date"),
            "end_date": storm_da.attrs.get("end_date"),
            "basin": storm_da.attrs.get("basin"),
            "category": storm_da.attrs.get("category"),

            # Exposure definition
            "description": (
                "Per-storm yearly exposure hours per pixel "
                f"where wind speed ≥ {wind_threshold} m/s"
            ),
            "definition": (
                "Exposure hours are computed as the number of timesteps "
                "with wind speed above the threshold. Each timestep "
                "is assumed to represent one hour."
            ),

            # Units & aggregation
            "units": "hours",
            "aggregation": "yearly",
            "wind_threshold_m_s": wind_threshold,
        })

        # Remove timestep-specific attrs if present
        yearly_exposure.attrs.pop("time_step", None)

        exposure_per_storm.append(yearly_exposure)

    return exposure_per_storm


######################################
#       Track Storm Duration         #
######################################
def track_storm_duration(
    storm_list_intensity: list[xr.DataArray],
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
) -> pd.DataFrame:
    """
    Track storm duration in days for each storm.

    Parameters
    ----------
    storm_list_speed : list[xr.DataArray]
        List of storm wind speed DataArrays with dims ('time', 'lat', 'lon').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - storm_name
        - storm_id
        - start_date
        - end_date
        - duration_days
    """

    records = []

    for storm_da in storm_list_intensity:
        storm_name = storm_da.attrs.get("storm_name")
        storm_id = storm_da.attrs.get("storm_id")
        start_date = storm_da.attrs.get("start_date")
        end_date = storm_da.attrs.get("end_date")

        # calculate duration in days
        start_dt = datetime.fromisoformat(str(start_date))
        end_dt = datetime.fromisoformat(str(end_date))
        duration_days = (end_dt - start_dt).days + 1  # inclusive

        # if duration is 30 days or more, log a warning
        if duration_days >= 30:
            records.append({
                "storm_name": storm_name,
                "storm_id": storm_id,
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": duration_days,
            })

    df_duration = pd.DataFrame(records)

    # save log
    if not df_duration.empty:
        log_path = LOG_DIR / f"storm_duration_{source_id}_{variant_label}_{experiment_id}_{batch_year}_{basin}_draw{draw}.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not log_path.exists()
        df_duration.to_csv(
            log_path,
            mode="a",
            header=write_header,
            index=False,
        )



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
        start_date = event.start_date
        end_date = event.end_date


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
            "start_date": start_date,
            "end_date": end_date,
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
#  Per Storm Days Impact Functions   #
######################################

def log_storm_year_split(
    log_path: Path,
    row: dict,
):
    """
    Append a storm calendar-year split record to a CSV log.

    Parameters
    ----------
    log_path : Path
        Path to the CSV log file.
    row : dict
        Dictionary of storm metadata describing a year split.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([row])

    write_header = not log_path.exists()
    df.to_csv(
        log_path,
        mode="a",
        header=write_header,
        index=False,
    )

def calculate_days_impact_single_storm_yearly(
    da_speed: xr.DataArray,
    next_da_speed: xr.DataArray | None,
    next_start_date: datetime | None,
    max_impact_days: int = 20,
    index: int | None = None,
) -> xr.DataArray:

    # ------------------------------
    # 1. Metadata
    # ------------------------------
    storm_name = da_speed.attrs.get("storm_name")
    storm_id = da_speed.attrs.get("storm_id")
    storm_basin = da_speed.attrs.get("basin")
    storm_category = da_speed.attrs.get("category")


    storm_start = da_speed.attrs.get("start_date")
    if storm_start is None:
        raise ValueError("Storm speed DataArray missing start_date")

    storm_start = datetime.fromisoformat(str(storm_start))
    storm_end = da_speed.attrs.get("end_date")

    nominal_end = storm_start + timedelta(days=max_impact_days)

    # ------------------------------
    # 2. Initial impact days
    # ------------------------------
    affected = (da_speed > 0).any(dim="time")

    impact_days_total = xr.zeros_like(
        affected,
        dtype=np.int16,
    )

    impact_days_total = xr.where(
        affected,
        max_impact_days,
        impact_days_total,
    )

    # ------------------------------
    # 3. Optional truncation
    # ------------------------------
    if (
        next_da_speed is not None
        and next_start_date is not None
        and next_start_date < nominal_end
    ):
        next_affected = (next_da_speed > 0).any(dim="time")
        overlapping = affected & next_affected

        if overlapping.any():
            delta_days = max(0, (next_start_date - storm_start).days)

            impact_days_total = xr.where(
                overlapping,
                np.minimum(impact_days_total, delta_days),
                impact_days_total,
            )

    # ------------------------------
    # 4. Split impact by calendar year
    # ------------------------------
    yearly_arrays: list[xr.DataArray] = []

    remaining_days = impact_days_total.copy()
    current_date = storm_start

    while remaining_days.max() > 0:
        year = current_date.year

        end_of_year = datetime(year, 12, 31)
        days_this_year = (end_of_year - current_date).days + 1

        slice_days = xr.where(
            remaining_days > 0,
            np.minimum(remaining_days, days_this_year),
            0,
        )

        da_year = slice_days.expand_dims(time=[year])
        yearly_arrays.append(da_year)

        remaining_days = (remaining_days - slice_days).clip(min=0)
        current_date = datetime(year + 1, 1, 1)

    # ------------------------------
    # 5. Handle zero-impact storms
    # ------------------------------
    if len(yearly_arrays) == 0:
        print(
            f"ℹ️ Storm {storm_name} has zero impact days "
            f"(index={index}, start={storm_start.date()})"
        )

        # Create an explicit zero-impact year (start year)
        impact_days_yearly = xr.zeros_like(
            impact_days_total,
            dtype=np.int16,
        ).expand_dims(time=[storm_start.year])

    else:
        impact_days_yearly = xr.concat(yearly_arrays, dim="time")

    # ------------------------------
    # 6. Metadata
    # ------------------------------
    impact_days_yearly.name = "days_impact"

    impact_days_yearly.attrs = {
        "description": "Per-storm pixel-level impact duration split by calendar year",
        "definition": (
            "Number of days a pixel is considered impacted by this storm per year. "
            "Total impact is capped at `max_impact_days` and truncated if a subsequent "
            "storm affects the same pixel. Impact days spanning calendar years are "
            "carried over into subsequent yearly slices."
        ),
        "units": "days",
        "storm_name": storm_name,
        "storm_id": storm_id,
        "basin": storm_basin,
        "category": storm_category,
        "start_date": storm_start.isoformat(),
        "end_date": storm_end,
        "max_impact_days": max_impact_days,
        "has_impact": len(yearly_arrays) > 0,
    }

    # ------------------------------
    # 7. Calendar year split logging
    # ------------------------------
    effective_days = int(impact_days_total.max())
    effective_end = storm_start + timedelta(days=effective_days)

    splits_calendar_year = effective_end.year > storm_start.year

    batch_start_year, batch_end_year = map(int, batch_year.split("-"))
    last_batch_year_exceeded = effective_end.year > batch_end_year

    if splits_calendar_year:
                
        # log the overlap, both current and next storm info
        log_dir = LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / "storm_split_calendar_year.csv"
        
        log_storm_year_split(
            log_path,
            {
                "storm_id": storm_id,
                "basin": storm_basin,
                "storm_start_date": storm_start.isoformat(),
                "storm_end_date": storm_end.isoformat(),
                "effective_end_date": effective_end.isoformat(),
                "start_year": storm_start.year,
                "end_year": effective_end.year,
                "batch_year": batch_year,
                "batch_end_year": batch_end_year,
                "last_batch_year_exceeded": last_batch_year_exceeded,
                "max_impact_days": max_impact_days,
                "effective_impact_days": effective_days,
            }
        )

    return impact_days_yearly


#############################
#       Helper Functions    #
#############################
def storm_start_from_dataset(ds: xr.Dataset) -> tuple:
    """
    Primary key: start_date
    Secondary key: sid (stable tie-breaker)
    """
    start = ds.attrs.get("start_date")
    if start is None:
        # absolute fallback
        start = ds.time.values[0]

    return (
        datetime.fromisoformat(str(start)),
        ds.attrs.get("sid", -1),
    )

def get_ordered_storm_indices(track_store: Path) -> list[int]:
    """
    Discover storms in a draw-level tracks Zarr store and return
    storm indices ordered chronologically using storm_start logic.
    """
    root = zarr.open_group(track_store, mode="r")

    storm_entries = []

    for group_name in root.group_keys():
        if not group_name.startswith("storm_"):
            continue

        try:
            storm_index = int(group_name.split("_")[1])
        except ValueError:
            continue

        # Open storm lazily via xarray (attrs only + time coord)
        ds = xr.open_zarr(track_store / group_name, consolidated=False)

        key = storm_start_from_dataset(ds)

        storm_entries.append(
            (storm_index, key)
        )

        # Explicitly close to avoid open file handles
        ds.close()

    if not storm_entries:
        raise RuntimeError(f"No storm groups found in {track_store}")

    # Sort by (start_date, sid)
    storm_entries_sorted = sorted(storm_entries, key=lambda x: x[1])

    # Extract ordered indices
    return [idx for idx, _ in storm_entries_sorted]



def sanitize_attrs(attrs: dict) -> dict:
    """
    Recursively sanitize a dictionary of attributes to be Zarr v3 compatible.
    Converts all non-JSON-serializable types to native types or strings.
    """
    safe = {}
    for k, v in attrs.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        elif isinstance(v, (np.integer, np.floating)):
            safe[k] = v.item()
        elif isinstance(v, (np.bool_)):
            safe[k] = bool(v)
        elif isinstance(v, np.datetime64):
            safe[k] = str(v)
        elif isinstance(v, (list, tuple)):
            safe[k] = [
                x.item() if isinstance(x, (np.integer, np.floating)) else
                bool(x) if isinstance(x, np.bool_) else
                str(x)
                for x in v
            ]
        elif isinstance(v, dict):
            safe[k] = sanitize_attrs(v)
        else:
            # last-resort stringify for unknown types
            safe[k] = str(v)
    return safe

def read_storm_metadata(
    track_store: Path,
    storm_index: int,
) -> dict:
    """
    Read minimal metadata for a single storm without loading full arrays.
    """
    storm_key = f"storm_{storm_index:04d}"
    ds = xr.open_zarr(track_store, group=storm_key, consolidated=False)

    start_date = pd.Timestamp(ds.time.values[0]).to_pydatetime()
    end_date = pd.Timestamp(ds.time.values[-1]).to_pydatetime()

    if start_date is None:
        raise ValueError(f"Storm {storm_index} missing start_date")

    return {
        "start_date": datetime.fromisoformat(str(start_date)),
        "end_date": (
            datetime.fromisoformat(str(end_date))
            if end_date is not None
            else None
        ),
        "storm_id": ds.attrs.get("storm_id"),
        "storm_name": ds.attrs.get("storm_name"),
        "basin": ds.attrs.get("basin"),
        "category": ds.attrs.get("category"),
    }

def log_storm_overlap(
    log_path: Path,
    row: dict,
):
    df = pd.DataFrame([row])

    write_header = not log_path.exists()
    df.to_csv(
        log_path,
        mode="a",
        header=write_header,
        index=False,
    )


def check_next_storm(
    i: int,
    storm_indices: list[int],
    track_store: Path,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    storm_list_speed: list[xr.DataArray],
    centroids,
    max_impact_days: int = 20,
) -> tuple[xr.DataArray | None, int | None]:
    """
    Determine if the next storm overlaps the current storm's impact period,
    and if so, load the next storm's wind speed for truncation calculations.

    Returns
    -------
    next_storm_speed : xr.DataArray | None
        The wind speed array of the next storm if truncation is needed.
    next_storm_index : int | None
        Index of the next storm if it exists, else None.
    """
    if i + 1 < len(storm_indices):
        next_storm_index = storm_indices[i + 1]
        print(f"Next storm index: {next_storm_index}")

        # Read metadata of the next storm
        next_storm_metadata = read_storm_metadata(track_store, next_storm_index)
        next_start_date = next_storm_metadata["start_date"]

        # Compute nominal end of current storm
        current_storm_end_nominal = pd.Timestamp(
            storm_list_speed[0].attrs["start_date"]
        ).to_pydatetime() + timedelta(days=max_impact_days)

        # Check if next storm occurs before current storm's nominal end
        if next_start_date < current_storm_end_nominal:
            print("Next storm overlaps current storm → need next speed array")

            # log the overlap, both current and next storm info
            log_dir = LOG_DIR
            log_dir.mkdir(parents=True, exist_ok=True)

            log_path = log_dir / "storm_overlaps.csv"

            overlap_duration_days = (
                current_storm_end_nominal - next_start_date
            ).days


            log_storm_overlap(log_path, {
                "source_id": source_id,
                "variant_label": variant_label,
                "experiment_id": experiment_id,
                "batch_year": batch_year,
                "basin": basin,
                "draw": draw,
                "current_storm_index": storm_indices[i],
                "current_storm_start_date": storm_list_speed[0].attrs["start_date"],
                "current_storm_end_date": storm_list_speed[0].attrs["end_date"],
                "next_storm_index": next_storm_index,
                "next_storm_start_date": next_start_date.isoformat(),
                "next_storm_end_date": next_storm_metadata["end_date"],
                "overlap_duration_days": overlap_duration_days,
            })

            # Load next storm
            next_tc_tracks = prepare_zarr_files(
                source_id=source_id,
                variant_label=variant_label,
                experiment_id=experiment_id,
                batch_year=batch_year,
                basin=basin,
                draw=draw,
                storm_index=next_storm_index,
            )
            next_haz = generate_hazard_per_track(next_tc_tracks, centroids)
            next_storm_speed = generate_speed_per_storm(next_haz, centroids, next_tc_tracks)
        else:
            next_storm_speed = None
    else:
        next_storm_index = None
        next_storm_speed = None

    return next_storm_speed, next_storm_index

#######################################
#    Save Per Storm Functions         #
#######################################

def save_single_storm_intensity(
    da: xr.DataArray,
    storm_index: int,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    save_root: Path = SAVE_ROOT,
):
    """
    Save a single storm's intensity DataArray to a draw-level Zarr store.

    Parameters
    ----------
    da : xr.DataArray
        Single storm intensity.
    storm_index : int
        Index of the storm (used for Zarr group key).
    source_id, variant_label, experiment_id, batch_year, basin, draw : str/int
        Metadata for path construction.
    save_root : Path, optional
        Root path for saving.
    """
    save_root.mkdir(parents=True, exist_ok=True)

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    draw_store = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / "intensity"
        / f"intensity_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
    )

    draw_store.parent.mkdir(parents=True, exist_ok=True)

    # Use storm index for Zarr group key
    storm_key = f"storm_{storm_index:04d}"  # 4 digits to match original 
    
    # --- check if Zarr store exists and if this storm is already written ---
    if draw_store.exists():
        z = zarr.open(draw_store, mode="r")
        if storm_key in z:
            print(f"⚠️ Storm {storm_index} already exists in {draw_store}, skipping.")
            return
        
    # Defensive copy & cast
    da = da.copy()
    da.name = "intensity"
    if isinstance(da.attrs.get("basin"), xr.DataArray):
        da.attrs["basin"] = str(da.attrs["basin"].values[0])
    if da.dtype != "float32":
        da = da.astype("float32")

    # Chunk before converting to Dataset
    da = da.chunk({"lat": 50, "lon": 50})
    ds = da.to_dataset()
    ds.attrs.update(sanitize_attrs(da.attrs))

    # Encoding
    encoding = {
        "intensity": {
            "compressors": [
                {
                    "name": "blosc",
                    "configuration": {
                        "cname": "zstd",
                        "clevel": 3,
                        "shuffle": "shuffle",
                    },
                }
            ],
            "dtype": "float32",
            "fill_value": 0.0,
        }
    }

    # Always append to Zarr store; create if it doesn't exist
    ds.to_zarr(
        draw_store,
        group=storm_key,
        mode="a",
        encoding=encoding,
        zarr_format=3,
        consolidated=False,
    )

    # chmod_recursive(draw_store, mode=0o775)



def save_single_storm_exposure(
    da: xr.DataArray,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    storm_index: int,
    save_root: Path = SAVE_ROOT,
):
    """
    Save a single storm's exposure_hours DataArray to a Zarr store.

    If the Zarr store doesn't exist, it will be created automatically.
    """
    save_root.mkdir(parents=True, exist_ok=True)

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    draw_store = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / "exposure_hours"
        / f"exposure_hours_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
    )

    draw_store.parent.mkdir(parents=True, exist_ok=True)

    storm_key = f"storm_{storm_index:04d}"  # 4 digits to match original 

    # --- check if Zarr store exists and if this storm is already written ---
    if draw_store.exists():
        z = zarr.open(draw_store, mode="r")
        if storm_key in z:
            print(f"⚠️ Storm {storm_index} already exists in {draw_store}, skipping.")
            return
        
    # Defensive copy
    da = da.copy()
    da.name = "exposure_hours"

    # Ensure float32
    if da.dtype != "float32":
        da = da.astype("float32")

    # Chunk DataArray
    da = da.chunk({"lat": 50, "lon": 50})

    # Convert to Dataset
    ds = da.to_dataset()

    # Promote DataArray attrs to Dataset
    ds.attrs.update(sanitize_attrs(da.attrs))

    # Zarr encoding
    compressor = {
        "name": "blosc",
        "configuration": {
            "cname": "zstd",
            "clevel": 3,
            "shuffle": "shuffle",
        },
    }
    encoding = {
        "exposure_hours": {
            "compressors": [compressor],
            "dtype": "float32",
            "fill_value": 0.0,
        }
    }

    # Save to Zarr
    ds.to_zarr(
        draw_store,
        group=storm_key,
        mode="a",  # append to existing store or create if missing
        encoding=encoding,
        zarr_format=3,
        consolidated=False,
    )

    # chmod_recursive(draw_store, mode=0o775)


def save_single_storm_days_impact(
    da: xr.DataArray,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    storm_index: int,
    save_root: Path = SAVE_ROOT,
):
    """
    Save a single storm's days_impact DataArray to a Zarr store.

    If the Zarr store doesn't exist, it will be created automatically.
    """
    save_root.mkdir(parents=True, exist_ok=True)

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    draw_store = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / "days_impact"
        / f"days_impact_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
    )

    draw_store.parent.mkdir(parents=True, exist_ok=True)

    storm_key = f"storm_{storm_index:04d}"  # 4 digits to match original 

    # --- check if Zarr store exists and if this storm is already written ---
    if draw_store.exists():
        z = zarr.open(draw_store, mode="r")
        if storm_key in z:
            print(f"⚠️ Storm {storm_index} already exists in {draw_store}, skipping.")
            return
        
    # Defensive copy
    da = da.copy()
    da.name = "days_impact"

    # Ensure int16 for compact storage
    if da.dtype != "int16":
        da = da.astype("int16")

    # Chunking
    da = da.chunk({"lat": 50, "lon": 50})

    # Convert to Dataset
    ds = da.to_dataset()

    # Promote metadata to Dataset-level attrs
    ds.attrs.update(sanitize_attrs(da.attrs))

    # Zarr encoding
    compressor = {
        "name": "blosc",
        "configuration": {
            "cname": "zstd",
            "clevel": 3,
            "shuffle": "shuffle",
        },
    }
    encoding = {
        "days_impact": {
            "compressors": [compressor],
            "dtype": "int16",
            "fill_value": 0,
        }
    }

    # Save to Zarr (append to existing or create new)
    ds.to_zarr(
        draw_store,
        group=storm_key,
        mode="a",
        encoding=encoding,
        zarr_format=3,
        consolidated=False,
    )

    # chmod_recursive(draw_store, mode=0o775)


############################################
#              Main                        #
############################################
        
def process_single_storm(storm_info):

    """
    Process a single storm: compute intensity, speed, exposure, and days impact.
    This function is intended for parallel execution over storm indices.

    Parameters
    ----------
    storm_info : tuple
        (storm_index, track_store, storm_indices, centroids, save_root,
         source_id, variant_label, experiment_id, batch_year, basin, draw)
    """
    (
        i, storm_index, track_store, storm_indices, save_root,
        source_id, variant_label, experiment_id, batch_year, basin, draw
    ) = storm_info
    centroids = generate_basin_centroids(basin, res=RESOLUTION)
    # Load current storm
    tc_tracks = prepare_zarr_files(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        storm_index=storm_index,
    )

    haz = generate_hazard_per_track(tc_tracks, centroids)

    storm_list_intensity = generate_intensity_per_storm(haz, centroids, tc_tracks)

    # track storm duration for logging
    track_storm_duration(
        storm_list_intensity[0],
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    )

    # save storm intensity
    save_single_storm_intensity(
        storm_list_intensity[0],
        storm_index=storm_index,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        save_root=save_root,
    )
    del storm_list_intensity

    storm_list_speed = generate_speed_per_storm(haz, centroids, tc_tracks)
    storm_list_exposure = compute_yearly_exposure_per_storm(
        storm_list_speed,
        wind_threshold=17.0,
    )

    # save exposure
    save_single_storm_exposure(
        storm_list_exposure[0],
        storm_index=storm_index,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        save_root=save_root,
    )
    del storm_list_exposure


    next_storm_speed, next_storm_index = check_next_storm(
        i=i,
        storm_indices=storm_indices,
        track_store=track_store,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        storm_list_speed=storm_list_speed,
        centroids=centroids,
        max_impact_days=20,
    )


    days_impact = calculate_days_impact_single_storm_yearly(
        da_speed=storm_list_speed[0],
        next_da_speed=next_storm_speed[0] if next_storm_speed else None,
        next_start_date=read_storm_metadata(track_store, next_storm_index)["start_date"] if next_storm_speed else None,
        max_impact_days=20,
    )

    # save days impact
    save_single_storm_days_impact(
        days_impact,
        storm_index=storm_index,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        save_root=save_root,
    )
    del days_impact, storm_list_speed
    del tc_tracks, haz, next_storm_speed
    gc.collect()

def main(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    save_root: Path = SAVE_ROOT,
):
    """
    Main pipeline for storm processing.

    Parameters
    ----------
    storm_type : str
        "direct" → full pipeline (intensity + exposure + days impact)
        "indirect" → only intensity
    """

    # step 1: get draw level tracks
    track_store = read_custom_tracks(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    )


    storm_indices = get_ordered_storm_indices(track_store)

    storm_args = [
        (i, idx, track_store, storm_indices, save_root,
        source_id, variant_label, experiment_id, batch_year, basin, draw)
        for i, idx in enumerate(storm_indices)
    ]

    results = run_parallel(
        runner=process_single_storm,
        arg_list=storm_args,
        num_cores=8,
    )

    draw_store = (
        save_root
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
    )
    metrics = ["intensity", "exposure_hours", "days_impact"]

    for metric in metrics:
        metric_store = draw_store / metric
        draw_text = "" if draw == 0 else f"_e{draw - 1}"
        start_year, end_year = batch_year.split("-")
        final_zarr_path = (
            metric_store
            / f"{metric}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}_final.zarr"
        )

        try:
            chmod_recursive(final_zarr_path, mode=0o775)
        except Exception as e:
            print(f"⚠️ Could not set permissions for {final_zarr_path}: {e}")

    print(f"✅ Completed processing for draw {draw} of basin {basin}.")




main(
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
    batch_year=batch_year,
    basin=basin,
    draw=draw,
)

"""
103 total source_id, variant_label, experiment_id, batch_year combination parameters
7 total basins
250 draws per combination
In total we have 103 * 7 * 250 = 180250 unique runs to process
We use multithreading / multiprocessing to parallelize each task over thier given storms


Single draw with 8 cores takes approximately 1-15 minutes and 15-25GB of memory

| Concurrent Tasks Running | Waves Needed | Total Runtime | Total Runtime (hours) | Total Runtime (days) |
| ------------------------ | ------------ | ------------- | --------------------- | -------------------- |
| 100                      | 1,803        | 27,045 min    | 451 hrs               | 18.8 days            |
| 250                      | 721          | 10,815 min    | 180 hrs               | 7.5 days             |
| 500                      | 361          | 5,415 min     | 90 hrs                | 3.8 days             |
| 1,000                    | 181          | 2,715 min     | 45 hrs                | 1.9 days             |
| 2,000                    | 91           | 1,365 min     | 22.8 hrs              | 0.95 days            |
| 3,000                    | 61           | 915 min       | 15.3 hrs              | 0.64 days            |
| 5,000                    | 37           | 555 min       | 9.3 hrs               | 0.39 days            |

"""