"""
Stage 4A resource-assignment step: consume the per-(storm × admin) metadata
from every completed stage 4A task and produce the resource-estimation
parquet that stage 4B's launcher consumes. Runs as a downstream Jobmon
task in the stage 4A workflow, with all per-(combo × draw) metadata tasks
as upstream dependencies.

Three steps:
  1. enumerate_paths()      — build the list of expected admin_level_metadata
                              parquets, one per (combo × draw).
  2. compile_metadata(...)  — parallel-read + concat into one compiled parquet
                              (checkpointed so step 3 can be re-run on its own).
  3. assign_resources(...)  — bin (storm × location) tasks into resource size
                              classes and save the launcher-ready parquet.

Resource sizing is driven by **two** stage-4A columns only:
  - `area_100m2`  → n_pixels → size_class → (memory_gb, runtime_min)
  - `location_id == LOC_ID_OVERRIDE_VERY_LARGE` → force "very_large"
All other 4A columns are carried through but unconsulted.

Resource map version: v14
  - 3-min runtimes collapsed → 2 min across the map
  - extreme bucket runtime 4 → 6 min
  - loc_id == 22 forced into "very_large" (29GB / 4 min) regardless of pixel count
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rra_tools.parallel import run_parallel  # type: ignore
from rasterra import RasterArray  # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument("--admin_level", type=int, default=0, choices=[0, 1])
args = parser.parse_args()
ADMIN_LEVEL = args.admin_level

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
CLIMADA_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada")
SAVE_ROOT = CLIMADA_ROOT / "output" / f"stage4a_metadata_admin{ADMIN_LEVEL}"
META_CSV = CLIMADA_ROOT / "input" / "cmip6" / "level_4_task_assignments.csv"

DRAWS = list(range(100))
BASELINE_BATCH_YEAR = "1965-1969"  # historical baseline; excluded from stage 4

PIXEL_SIZE_M = 100
PIXEL_AREA_M2 = PIXEL_SIZE_M ** 2  # 100 m × 100 m = 10,000 m²

# loc_id 22 = India at admin0; empirically heavier than pixel count suggests.
# Not applicable at admin1 (different location_id space).
LOC_ID_OVERRIDE_VERY_LARGE = 22 if ADMIN_LEVEL == 0 else None

COMPILED_PARQUET = SAVE_ROOT / "compiled_admin_level_metadata.parquet"
OUTPUT_PARQUET = SAVE_ROOT / "resource_estimation_all_storms.parquet"

CHUNK_SIZE = 1500   # parallel-read chunk size; 500-2000 is the typical sweet spot
NUM_CORES = 40


# Resource map v14
PIXEL_BINS = [0, 1e6, 2e7, 2e8, 5e8, 1.5e9, np.inf]
SIZE_CLASSES = ["very_small", "small", "medium", "large", "very_large", "extreme"]
RESOURCE_MAP = {
    "very_small": {"mem_gb": 3,  "runtime_min": 1},
    "small":      {"mem_gb": 4,  "runtime_min": 1},
    "medium":     {"mem_gb": 5,  "runtime_min": 2},
    "large":      {"mem_gb": 10, "runtime_min": 2},
    "very_large": {"mem_gb": 29, "runtime_min": 4},
    "extreme":    {"mem_gb": 70, "runtime_min": 6},
}

# Defensive runtime re-bin. The values RESOURCE_MAP emits are {1, 2, 4, 6};
# they all map to themselves. The `8` bucket is a safety ceiling that should
# only fire if RESOURCE_MAP gains a new entry > 6 min.
RUNTIME_BINS = [0, 1, 2, 4, 6, np.inf]
RUNTIME_LABELS = [1, 2, 4, 6, 8]


# ----------------------------------------------------------------------
# Path helper — single source of truth, matches save_draw_dataframe in
# 04_admin_level_exposure_a_main.py
# ----------------------------------------------------------------------
def _admin_metadata_path(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
) -> Path:
    """Path to a single (combo × draw)'s admin_level_metadata parquet."""
    start_year, end_year = batch_year.split("-")
    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    return (
        SAVE_ROOT
        / source_id / variant_label / experiment_id / batch_year / basin
        / f"tc_risk_draw_{draw}" / "admin_level_metadata"
        / (
            f"admin_level_metadata_{basin}_{source_id}_{variant_label}_"
            f"{experiment_id}_{start_year}01_{end_year}12{draw_text}.parquet"
        )
    )


# ----------------------------------------------------------------------
# Step 1 — enumerate expected paths
# ----------------------------------------------------------------------
def enumerate_paths() -> list[tuple[Path, int]]:
    """
    Cross-join level_4 task assignments with draws 0..99 and return a list
    of `(parquet_path, draw)` pairs — every (combo × draw) we expect 4A
    to have produced.
    """
    meta = pd.read_csv(META_CSV).drop(columns=["task_id", "draw"]).drop_duplicates()
    meta["basin"] = meta["basin"].fillna("NA")
    meta = meta.rename(columns={
        "model": "source_id",
        "variant": "variant_label",
        "scenario": "experiment_id",
        "time_period": "batch_year",
    })
    meta = meta[meta["batch_year"] != BASELINE_BATCH_YEAR].reset_index(drop=True)

    full_tasks = (
        meta
        .assign(key=1)
        .merge(pd.DataFrame({"draw": DRAWS, "key": 1}), on="key")
        .drop(columns=["key"])
    )

    return [
        (
            _admin_metadata_path(
                source_id=row.source_id,
                variant_label=row.variant_label,
                experiment_id=row.experiment_id,
                batch_year=row.batch_year,
                basin=row.basin,
                draw=int(row.draw),
            ),
            int(row.draw),
        )
        for row in full_tasks.itertuples(index=False)
    ]


# ----------------------------------------------------------------------
# Step 2 — parallel read + concat
# ----------------------------------------------------------------------
def _read_parquet_chunk(path_draw_pairs: list[tuple[Path, int]]) -> pd.DataFrame:
    """Read a chunk of parquets and stamp each row with its draw number.
    Missing files are logged and skipped — runs safely against a partially-
    completed 4A pipeline."""
    dfs = []
    for path, draw in path_draw_pairs:
        try:
            df = pd.read_parquet(path)
        except FileNotFoundError:
            print(f"⚠️ Missing 4A metadata parquet: {path}")
            continue
        df["draw"] = draw
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _chunk(seq: list, chunk_size: int):
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]


def compile_metadata(path_draw_pairs: list[tuple[Path, int]]) -> pd.DataFrame:
    """Parallel-read every (path, draw) pair, concat into one DataFrame, and
    checkpoint to COMPILED_PARQUET so the resource step can be re-run on
    its own (e.g., to retune RESOURCE_MAP without redoing the I/O)."""
    chunks = list(_chunk(path_draw_pairs, CHUNK_SIZE))
    dfs = run_parallel(
        runner=_read_parquet_chunk,
        arg_list=chunks,
        num_cores=NUM_CORES,
        progress_bar=True,
    )
    final = pd.concat(dfs, ignore_index=True)
    final.to_parquet(COMPILED_PARQUET, index=False)
    print(f"Saved compiled metadata → {COMPILED_PARQUET} ({len(final):,} rows)")
    return final


# ----------------------------------------------------------------------
# Step 3 — assign resources
# ----------------------------------------------------------------------
def assign_resources(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin every (storm × location) row by `n_pixels` (derived from 4A's
    `area_100m2`) into one of six size classes, map each to a
    (memory_gb, runtime_min) budget, and return one row per task ready
    for stage 4B's launcher.

    Override:
      - `location_id == LOC_ID_OVERRIDE_VERY_LARGE` is forced into the
        "very_large" bucket regardless of n_pixels (empirically heavier
        than its pixel count suggests).
    """
    meta_df = meta_df.copy()
    meta_df["n_pixels"] = meta_df["area_100m2"] / PIXEL_AREA_M2

    meta_df["size_class"] = pd.cut(
        meta_df["n_pixels"], bins=PIXEL_BINS, labels=SIZE_CLASSES
    )

    mem_lookup = {k: v["mem_gb"] for k, v in RESOURCE_MAP.items()}
    time_lookup = {k: v["runtime_min"] for k, v in RESOURCE_MAP.items()}

    # Cast categorical → str so the loc_id override below can rewrite it.
    meta_df["size_class"] = meta_df["size_class"].astype(str)
    meta_df["memory_gb"] = meta_df["size_class"].map(mem_lookup).astype(int)
    meta_df["runtime_min"] = meta_df["size_class"].map(time_lookup).astype(int)

    # loc_id override (admin0 only)
    if LOC_ID_OVERRIDE_VERY_LARGE is not None:
        loc_mask = meta_df["location_id"] == LOC_ID_OVERRIDE_VERY_LARGE
        if loc_mask.any():
            very_large = RESOURCE_MAP["very_large"]
            print(
                f"[INFO] Overriding {loc_mask.sum():,} loc_id="
                f"{LOC_ID_OVERRIDE_VERY_LARGE} tasks → very_large "
                f"({very_large['mem_gb']}GB/{very_large['runtime_min']}min)"
            )
            meta_df.loc[loc_mask, "memory_gb"] = very_large["mem_gb"]
            meta_df.loc[loc_mask, "runtime_min"] = very_large["runtime_min"]
            meta_df.loc[loc_mask, "size_class"] = "very_large"

    # Defensive runtime ceiling (see RUNTIME_BINS comment for rationale).
    meta_df["runtime_min_binned"] = pd.cut(
        meta_df["runtime_min"],
        bins=RUNTIME_BINS,
        labels=RUNTIME_LABELS,
    ).astype(int)

    # One row per (storm × location) task.
    task_cols = [
        "source_id", "variant_label", "experiment_id",
        "batch_year", "basin", "draw", "storm_id", "location_id",
    ]
    launcher_df = (
        meta_df[task_cols + ["n_pixels", "size_class", "memory_gb", "runtime_min_binned"]]
        .drop_duplicates(subset=task_cols)
        .reset_index(drop=True)
    )
    launcher_df["num_cores"] = 1

    # Sanity reporting
    print("\n=== Template count ===")
    print(
        launcher_df
        .groupby(["runtime_min_binned", "memory_gb"])
        .size()
        .reset_index(name="n_tasks")
        .sort_values(["memory_gb", "runtime_min_binned"])
        .to_string(index=False)
    )
    print(f"\nTotal storm-location tasks: {len(launcher_df):,}")
    print(
        "Unique templates:           "
        f"{launcher_df[['runtime_min_binned', 'memory_gb']].drop_duplicates().shape[0]}"
    )
    print("\n=== Runtime distribution ===")
    print(launcher_df["runtime_min_binned"].value_counts().sort_index())
    print("\n=== Size class distribution ===")
    print(launcher_df["size_class"].value_counts().sort_index())
    if LOC_ID_OVERRIDE_VERY_LARGE is not None:
        print(f"\n=== loc_id={LOC_ID_OVERRIDE_VERY_LARGE} override check ===")
        loc_subset = launcher_df[launcher_df["location_id"] == LOC_ID_OVERRIDE_VERY_LARGE]
        print(f"  Tasks:      {len(loc_subset):,}")
        if not loc_subset.empty:
            print(f"  Memory:     {loc_subset['memory_gb'].unique()}")
            print(f"  Runtime:    {loc_subset['runtime_min_binned'].unique()}")
            print(f"  Size class: {loc_subset['size_class'].unique()}")

    launcher_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved → {OUTPUT_PARQUET}")

    return launcher_df


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
if __name__ == "__main__":
    path_draw_pairs = enumerate_paths()
    compiled = compile_metadata(path_draw_pairs)

    # `compiled` is already in memory from compile_metadata above, but the
    # checkpoint at COMPILED_PARQUET means you can re-run just step 3 by
    # commenting out the two lines above and uncommenting the next:
    # compiled = pd.read_parquet(COMPILED_PARQUET)

    assign_resources(compiled)
