import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd  # type: ignore
import sys

metrics = ["days_impact", "intensity", "exposure_hours"]
MAX_WORKERS = 4


def build_draw_store_path(row, metric, save_root: Path):
    source_id = str(row["source_id"])
    variant_label = str(row["variant_label"])
    experiment_id = str(row["experiment_id"])
    batch_year = str(row["batch_year"])
    basin = str(row["basin"])
    draw = int(row["draw"])

    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    start_year, end_year = batch_year.split("-")

    return (
        save_root
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


def delete_row(row, save_root: Path):
    deleted = 0
    for metric in metrics:
        path = build_draw_store_path(row, metric, save_root)
        if path.exists():
            if delete_path(path):
                deleted += 1
    return deleted


def main(csv_path, save_root: Path):
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
    ).fillna("NA")

    total = len(df)
    deleted_total = 0

    print(f"Loaded {total} tasks")
    print(f"Deleting with {MAX_WORKERS} workers")
    print(f"Save root: {save_root}")
    print("Starting...\n")

    rows = df.to_dict("records")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, deleted in enumerate(
            ex.map(lambda r: delete_row(r, save_root), rows), 1
        ):
            deleted_total += deleted
            if i % 5 == 0:
                print(f"Processed {i}/{total} rows | deleted stores={deleted_total}")

    print("\nDone")
    print(f"Total stores deleted: {deleted_total}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python delete_zarr_from_tasks.py tasks.csv /path/to/save_root")
        sys.exit(1)

    csv_path = sys.argv[1]
    save_root = Path(sys.argv[2])

    main(csv_path, save_root)