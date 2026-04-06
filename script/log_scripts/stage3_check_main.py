from pathlib import Path
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
import os
import rasterra as rt # type: ignore
import pandas as pd  # type: ignore
from rasterio.features import shapes  # type: ignore
from shapely.geometry import shape  # type: ignore
import geopandas as gpd  # type: ignore
import argparse
import gc
import rasterio  # type: ignore
from rra_tools.parallel import run_parallel  # type: ignore
from shapely.geometry import box
import shapely
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString
from shapely.ops import split, unary_union
import pyarrow.parquet as pq  # type: ignore

parser = argparse.ArgumentParser(description="Run CLIMADA code")

# Define arguments
parser.add_argument("--storm_draw", type=str, required=True, help="Storm Draw")
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
parser.add_argument("--basin", type=str, required=True, help="Basin")
parser.add_argument("--relative_risk", type=str, required=True, help="Relative risk type")
parser.add_argument("--sample_name", type=str, required=True, help="Sample name for relative risk")


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


# Constants
PAF_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage2_v2/")
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage3_v2/")
LOG_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage3_v2_log/")

def classify_error(e: Exception) -> str:
    msg = str(e).lower()

    if isinstance(e, FileNotFoundError):
        return "missing_file"
    if "zero-byte" in msg:
        return "zero_byte"
    if "empty paf dataframe" in msg:
        return "empty_df"
    if "corrupt parquet" in msg or "arrowinvalid" in msg:
        return "corrupt_parquet"

    return "unknown"

def log_paf_error(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    year: int,
    relative_risk: str,
    sample_name: str,
    error_type: str,
    root: Path = LOG_ROOT,
):
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "storm_draw": storm_draw,
        "source_id": source_id,
        "variant_label": variant_label,
        "experiment_id": experiment_id,
        "batch_year": batch_year,
        "basin": basin,
        "year": year,
        "relative_risk": relative_risk,
        "sample_name": sample_name,
        "error_type": error_type,
    }

    df = pd.DataFrame([record])


    fname = f"{storm_draw}_{source_id}_{variant_label}_{experiment_id}_{batch_year}_{year}_{basin}_{relative_risk}_{sample_name}.parquet"
    df.to_parquet(root / fname, index=False)

def safe_check_and_log(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    year: int,
    relative_risk: str,
    sample_name: str,
    root: Path = LOG_ROOT,
) -> bool:
    try:
        return check_if_year_complete(
            storm_draw=storm_draw,
            source_id=source_id,
            variant_label=variant_label,
            sample_name=sample_name,
            relative_risk=relative_risk,
            experiment_id=experiment_id,
            batch_year=batch_year,
            year=year,
            basin=basin,
            save_root=SAVE_ROOT,
        )
    except Exception as e:
        error_type = classify_error(e)

        log_paf_error(
            storm_draw=storm_draw,
            source_id=source_id,
            variant_label=variant_label,
            experiment_id=experiment_id,
            batch_year=batch_year,
            basin=basin,
            year=year,
            relative_risk=relative_risk,
            sample_name=sample_name,
            error_type=error_type,
            root=root,
        )

        print(f"[ERROR] {error_type} | {e}")
        return False


def check_if_year_complete(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    sample_name: str,
    relative_risk: str,
    experiment_id: str,
    batch_year: str,
    year: int,
    basin: str,
    save_root: Path,
) -> bool:
    """Return True if yearly PAF parquet exists and is valid.
    
    Raises:
        FileNotFoundError
        ValueError
        RuntimeError
    """

    year = str(year)

    save_dir = (
        save_root
        / storm_draw
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / year
        / "paf_df"
    )

    start_year, end_year = batch_year.split("-")

    filename = (
        f"paf_{storm_draw}_{relative_risk}_{sample_name}_{basin}_"
        f"{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}.parquet"
    )

    save_path = save_dir / filename

    # ----------------------------
    # 1. Missing file
    # ----------------------------
    if not save_path.exists():
        raise FileNotFoundError(f"Missing PAF parquet: {save_path}")

    # ----------------------------
    # 2. Zero-byte file
    # ----------------------------
    if save_path.stat().st_size == 0:
        raise ValueError(f"Zero-byte parquet file: {save_path}")

    # ----------------------------
    # 3. Read parquet
    # ----------------------------
    try:
        pf = pq.ParquetFile(save_path)
        if pf.metadata.num_rows == 0:
            raise ValueError(f"Empty PAF dataframe: {save_path}")
    except Exception as e:
        raise RuntimeError(
            f"Corrupt parquet (read failed): {save_path} | {e}"
        ) from e



    return True

def check_batch_complete(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    sample_name: str,
    relative_risk: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    save_root: Path = SAVE_ROOT,
) -> bool:

    start_year, end_year = map(int, batch_year.split("-"))
    years = range(start_year, end_year + 1)

    for year in years:
        if not safe_check_and_log(
            storm_draw=storm_draw,
            source_id=source_id,
            variant_label=variant_label,
            experiment_id=experiment_id,
            batch_year=batch_year,
            basin=basin,
            year=year,
            relative_risk=relative_risk,
            sample_name=sample_name,
        ):
            return False

    return True
##########################################
#            MAIN FUNCTION               #
##########################################


def main(
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

    if check_batch_complete(
        storm_draw=storm_draw,
        source_id=source_id,
        variant_label=variant_label,
        sample_name=sample_name,
        relative_risk=relative_risk,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        save_root=save_root,
    ):
        print("Task already complete. Exiting.")
        return

    print("Missing years detected. Continue processing.")
    
main(
    storm_draw=storm_draw,
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
    batch_year=batch_year,
    basin=basin,
    relative_risk=relative_risk,
    sample_name=sample_name,
)
