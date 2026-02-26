from pathlib import Path
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

parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--draw", type=int, required=True, help="Draw index")
parser.add_argument("--storm_type", type=str, required=True, help="Direct or Indirect")

# Parse arguments
args = parser.parse_args()
source_id = args.source_id
variant_label = args.variant_label
experiment_id = args.experiment_id
batch_year = args.batch_year
basin = args.basin
draw = args.draw
storm_type = args.storm_type

# Constants
ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")
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

    # temp
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
        'SP': ['180E', '45S', '250E', '0S'], # Original SA - possible mismatch
        'WP': ['100E', '0N', '180E', '60N'],
        'GL': ['0E', '90S', '360E', '90N']
    }

    if basin not in basin_bounds:
        raise ValueError(f"Basin '{basin}' not recognized. Available: {list(basin_bounds.keys())}")

    def parse_coord(coord_str: str) -> float:
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
#  Per Storm Days Impact Functions   #
######################################
def calculate_days_impact_per_storm_yearly(
    storm_list_speed: list[xr.DataArray],
    max_impact_days: int = 20,
) -> list[xr.DataArray]:
    """
    Calculate per-storm, per-pixel impact days split by calendar year,
    with carry-over across years and truncation by subsequent storms.
    """

    n_storms = len(storm_list_speed)

    # ------------------------------
    # 1. Extract storm start dates
    # ------------------------------
    start_dates = []
    for da in storm_list_speed:
        start = da.attrs.get("start_date")
        if start is None:
            raise ValueError("Storm speed DataArray missing start_date")
        start_dates.append(datetime.fromisoformat(str(start)))

    nominal_end_dates = [
        start + timedelta(days=max_impact_days)
        for start in start_dates
    ]

    days_impact_yearly_list = []

    # ------------------------------
    # 2. Per-storm computation
    # ------------------------------
    for i, da_speed in enumerate(storm_list_speed):

        # ---- metadata ----
        storm_name = da_speed.attrs.get("storm_name")
        storm_id = da_speed.attrs.get("storm_id")
        storm_basin = da_speed.attrs.get("basin")
        storm_category = da_speed.attrs.get("category")
        storm_start_date = start_dates[i]
        storm_end_date = da_speed.attrs.get("end_date")

        # ----------------------------------
        # 2a. Affected pixels
        # ----------------------------------
        affected = (da_speed > 0).any(dim="time")

        impact_days_total = xr.zeros_like(
            affected,
            dtype=np.int16,
        )

        impact_days_total = impact_days_total.where(~affected, max_impact_days)

        # ----------------------------------
        # 2b. Truncate using later storms
        # ----------------------------------
        for j in range(i + 1, n_storms):

            next_start = start_dates[j]

            if next_start >= nominal_end_dates[i]:
                break

            next_da = storm_list_speed[j]
            next_affected = (next_da > 0).any(dim="time")

            overlapping = affected & next_affected
            if overlapping.sum() == 0:
                continue

            delta_days = (next_start - storm_start_date).days
            delta_days = max(0, delta_days)

            impact_days_total = xr.where(
                overlapping,
                np.minimum(impact_days_total, delta_days),
                impact_days_total,
            )

        # ----------------------------------
        # 2c. Split total impact by year
        # ----------------------------------
        yearly_arrays = []

        remaining_days = impact_days_total.copy()
        current_date = storm_start_date

        while remaining_days.max() > 0:
            year = current_date.year

            end_of_year = datetime(year, 12, 31)
            days_this_year = (end_of_year - current_date).days + 1

            slice_days = xr.where(
                remaining_days > 0,
                np.minimum(remaining_days, days_this_year),
                0,
            )

            da_year = slice_days.expand_dims(
                time=[year]
            )

            yearly_arrays.append(da_year)

            remaining_days = remaining_days - slice_days
            remaining_days = remaining_days.clip(min=0)

            current_date = datetime(year + 1, 1, 1)

        # ----------------------------------
        # 2d. Combine years + metadata
        # ----------------------------------
        impact_days_yearly = xr.concat(yearly_arrays, dim="time")
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
            "start_date": storm_start_date.isoformat(),
            "end_date": storm_end_date,
            "max_impact_days": max_impact_days,
        }

        days_impact_yearly_list.append(impact_days_yearly)

    return days_impact_yearly_list

#############################
#       Helper Functions    #
#############################

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



def save_storm_intensity_list_draw(
    storm_list_rr: list[xr.DataArray],
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    # save_root: Path,
):
    save_root = Path("/mnt/share/scratch/users/mfiking/climada_test/")
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

    if draw_store.exists():
        shutil.rmtree(draw_store)


    draw_store.parent.mkdir(parents=True, exist_ok=True)

    for i, da in enumerate(storm_list_rr):
        storm_key = f"storm_{i:03d}"
        
        # defensive copy
        da = da.copy()
        da.name = "intensity"
        
        # basin attr: DataArray(time) → single string
        if isinstance(da.attrs.get("basin"), xr.DataArray):
            da.attrs["basin"] = str(da.attrs["basin"].values[0])


        # cast
        if da.dtype != "float32":
            da = da.astype("float32")

        # ✅ chunk the DataArray, not the encoding
        da = da.chunk({"lat": 50, "lon": 50})

        # convert to Dataset
        ds = da.to_dataset()

        # ✅ sanitize BEFORE attaching
        ds.attrs.update(sanitize_attrs(da.attrs))


        encoding = {
            "intensity": {
                "compressors": [
                    {
                        "name": "blosc",
                        "configuration": {  # Zarr v3 expects a "configuration" dict
                            "cname": "zstd",   # compression algorithm
                            "clevel": 3,        # compression level
                            "shuffle": "shuffle"   # must be a string: "none", "byte", or "bit"
                        }
                    }
                ],
                "dtype": "float32",
                "fill_value": 0.0,  # use lowercase for Xarray/Zarr
            }
        }

        ds.to_zarr(
            draw_store,
            group=storm_key,
            mode="w" if i == 0 else "a",
            encoding=encoding,
            zarr_format=3,
            consolidated=False,
        )

        chmod_recursive(draw_store, mode=0o775)


def save_storm_exposure_list_draw(
    storm_list_exposure: list[xr.DataArray],
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
):
    save_root = Path("/mnt/share/scratch/users/mfiking/climada_test/")
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
    if draw_store.exists():
        shutil.rmtree(draw_store)

    draw_store.parent.mkdir(parents=True, exist_ok=True)

    # Compressor tuned for sparse exposure fields
    compressor = {
        "name": "blosc",
        "configuration": {
            "cname": "zstd",
            "clevel": 3,
            "shuffle": "shuffle",  # must be str: "noshuffle", "shuffle", "bitshuffle"
        },
    }

    for i, da in enumerate(storm_list_exposure):
        storm_key = f"storm_{i:03d}"

        # defensive copy
        da = da.copy()
        da.name = "exposure_hours"

        # ensure float32
        if da.dtype != "float32":
            da = da.astype("float32")

        # ✅ chunk the DataArray (not the encoding)
        da = da.chunk({"lat": 50, "lon": 50})

        # convert to Dataset
        ds = da.to_dataset()

        # promote storm metadata to dataset-level attrs
        ds.attrs.update(sanitize_attrs(da.attrs))

        # encoding (Zarr v3–compatible)
        encoding = {
            "exposure_hours": {
                "compressors": [compressor],
                "dtype": "float32",
                "fill_value": 0.0,  # lowercase fill_value works with xarray
            }
        }

        ds.to_zarr(
            draw_store,
            group=storm_key,
            mode="w" if i == 0 else "a",
            encoding=encoding,
            zarr_format=3,
            consolidated=False,  # consistent with intensity
        )

        chmod_recursive(draw_store, mode=0o775)

def save_days_impact_list_draw(
    days_impact_list: list[xr.DataArray],
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
):
    save_root = Path("/mnt/share/scratch/users/mfiking/climada_test/")
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
    if draw_store.exists():
        shutil.rmtree(draw_store)
        
    draw_store.parent.mkdir(parents=True, exist_ok=True)

    # Compressor tuned for sparse integer fields (Zarr v3)
    compressor = {
        "name": "blosc",
        "configuration": {
            "cname": "zstd",
            "clevel": 3,
            "shuffle": "shuffle",  # must be str: "noshuffle", "shuffle", "bitshuffle"
        },
    }

    for i, da in enumerate(days_impact_list):
        storm_key = f"storm_{i:03d}"

        # defensive copy
        da = da.copy()
        da.name = "days_impact"

        # ensure compact integer storage
        if da.dtype != "int16":
            da = da.astype("int16")

        # ✅ chunk the DataArray
        da = da.chunk({"lat": 50, "lon": 50})

        # convert to Dataset
        ds = da.to_dataset()

        # promote storm metadata to dataset-level attrs
        ds.attrs.update(sanitize_attrs(da.attrs))

        # encoding (Zarr v3–compatible)
        encoding = {
            "days_impact": {
                "compressors": [compressor],
                "dtype": "int16",
                "fill_value": 0,  # lowercase fill_value
            }
        }

        ds.to_zarr(
            draw_store,
            group=storm_key,
            mode="w" if i == 0 else "a",
            encoding=encoding,
            zarr_format=3,
            consolidated=False,
        )

        chmod_recursive(draw_store, mode=0o775)

############################################
#              Main                        #
############################################
def main(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    storm_type: str = "direct",  # "direct" or "indirect"
):
    """
    Main pipeline for storm processing.

    Parameters
    ----------
    storm_type : str
        "direct" → full pipeline (intensity + exposure + days impact)
        "indirect" → only intensity
    """

    # prepare TCTracks from Zarr files
    tc_tracks = prepare_zarr_files(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    )

    # generate basin centroids
    centroids = generate_basin_centroids(basin, res=RESOLUTION)

    # generate hazard object
    haz = generate_hazard_per_track(tc_tracks, centroids)

    # generate per-storm intensity
    storm_list_intensity = generate_intensity_per_storm(haz, centroids, tc_tracks)
    save_storm_intensity_list_draw(
        storm_list_intensity,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
    )
    del storm_list_intensity

    if storm_type == "direct":
        # generate per-storm wind speed
        storm_list_speed = generate_speed_per_storm(haz, centroids, tc_tracks)

        # calculate per-storm yearly exposure
        storm_list_exposure = compute_yearly_exposure_per_storm(
            storm_list_speed,
            wind_threshold=17.0,
        )
        save_storm_exposure_list_draw(
            storm_list_exposure,
            source_id=source_id,
            variant_label=variant_label,
            experiment_id=experiment_id,
            batch_year=batch_year,
            basin=basin,
            draw=draw,
        )
        del storm_list_exposure

        # generate per-storm days impact
        days_impact_list = calculate_days_impact_per_storm_yearly(
            storm_list_speed,
            max_impact_days=20,
        )
        save_days_impact_list_draw(
            days_impact_list,
            source_id=source_id,
            variant_label=variant_label,
            experiment_id=experiment_id,
            batch_year=batch_year,
            basin=basin,
            draw=draw,
        )
        del days_impact_list


main(
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
    batch_year=batch_year,
    basin=basin,
    draw=draw,
    storm_type=storm_type,
)
