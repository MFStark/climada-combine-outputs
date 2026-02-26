import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd  # type: ignore
import sys

SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage0_DELETE")
metrics = ["days_impact", "intensity", "exposure_hours"]

MAX_WORKERS = 32   # adjust if filesystem struggles


def build_draw_store_path(row, metric):
    source_id = str(row["source_id"])
    variant_label = str(row["variant_label"])
    experiment_id = str(row["experiment_id"])
    batch_year = str(row["batch_year"])
    basin = str(row["basin"])
    draw = int(row["draw"])

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    return (
        SAVE_ROOT
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / metric
        / f"{metric}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
    )


def delete_path(path: Path):
    try:
        subprocess.run(
            ["rm", "-rf", str(path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def delete_row(row):
    deleted = 0
    for metric in metrics:
        path = build_draw_store_path(row, metric)
        if path.exists():
            if delete_path(path):
                deleted += 1
    return deleted


def main(csv_path):
    df = pd.read_csv(
        csv_path,
        dtype={
            "source_id": "string",
            "variant_label": "string",
            "experiment_id": "string",
            "batch_year": "string",
            "basin": "string",
            "draw": "int64",
        }
    )
    # replace any nan with NA
    df = df.fillna("NA")

    total = len(df)
    deleted_total = 0

    print(f"Loaded {total} tasks")
    print(f"Deleting with {MAX_WORKERS} workers")
    print("Starting...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, deleted in enumerate(ex.map(delete_row, df.to_dict("records")), 1):
            deleted_total += deleted
            if i % 5 == 0:
                print(f"Processed {i}/{total} rows | deleted stores={deleted_total}")

    print("\nDone")
    print(f"Total stores deleted: {deleted_total}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_zarr_from_tasks.py tasks.csv")
        sys.exit(1)

    main(sys.argv[1])