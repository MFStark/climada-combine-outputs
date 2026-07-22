"""
Stage 4B post-processing step: consume every per-(storm × location_id)
parquet written by stage 4B and consolidate into one launcher-friendly
DataFrame.  Runs as a downstream Jobmon task in the stage 4B workflow,
with all per-(storm × loc) tasks as upstream dependencies.

Path enumeration uses 4A's `compiled_admin_level_metadata.parquet` (which
already carries the `year` column the 4B launcher loses) instead of
crawling the stage4b_v2 tree.  At 35M+ expected paths we never
materialize a list of `Path` objects — chunks are sliced directly from
the metadata DataFrame and each worker constructs path strings on the
fly while reading.

Two steps:
  1. enumerate_meta()        — load the 4A compiled metadata and project
                               down to the columns needed to build paths.
  2. compile_exposure(meta)  — parallel-read every expected 4B parquet in
                               row-chunks, concat into one DataFrame, and
                               write the consolidated parquet.
"""

import argparse
import pandas as pd
from pathlib import Path
import json
import time
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from rra_tools.parallel import run_parallel  # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument("--admin_level", type=int, default=0, choices=[0, 1])
args = parser.parse_args()
ADMIN_LEVEL = args.admin_level

CLIMADA_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada")
COMPILED_4A_PARQUET = (
    CLIMADA_ROOT / "output" / f"stage4a_metadata_admin{ADMIN_LEVEL}"
    / "compiled_admin_level_metadata.parquet"
)

SAVE_ROOT = CLIMADA_ROOT / "output" / f"stage4b_v2{'_admin1' if ADMIN_LEVEL == 1 else ''}"
CONSOLIDATED_DIR = SAVE_ROOT / "_consolidated"
CONSOLIDATED_PARQUET = CONSOLIDATED_DIR / "storm_exposure_all.parquet"
PARTS_DIR = CONSOLIDATED_DIR / "parts"
MANIFEST_JSONL = CONSOLIDATED_DIR / "parts_manifest.jsonl"

# Path-build columns lifted from 4A. `year` is the only one the 4B
# launcher's own meta_df drops (it's derived inside the worker from
# storm.start_date), so loading from the 4A compile is the cheapest way
# to recover it without re-opening every CMIP zarr.
ID_COLS = [
    "source_id", "variant_label", "experiment_id",
    "batch_year", "basin", "draw",
    "storm_id", "location_id", "year",
]

CHUNK_SIZE = 5000   # rows per parallel-read chunk; 35M / 5000 ≈ 7K chunks
NUM_CORES = 64
BATCH_CHUNKS = 64   # number of CHUNK_SIZE chunks processed in one wave
WRITE_MASTER_PARQUET = True


# ----------------------------------------------------------------------
# Path helper — must match `save_storm_exposure` in
# 04_admin_level_exposure_b_main.py exactly.
# ----------------------------------------------------------------------
def _build_4b_path(row) -> Path:
    start_year, end_year = row.batch_year.split("-")
    draw = int(row.draw)
    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    return (
        SAVE_ROOT
        / row.source_id / row.variant_label / row.experiment_id
        / row.batch_year / row.basin
        / f"tc_risk_draw_{draw}" / "storm_exposure" / str(int(row.year))
        / (
            f"storm_{row.storm_id}_loc_{row.location_id}_{row.basin}_"
            f"{row.source_id}_{row.variant_label}_{row.experiment_id}_"
            f"{start_year}01_{end_year}12{draw_text}.parquet"
        )
    )


# ----------------------------------------------------------------------
# Step 1 — enumerate (storm × admin × draw) inventory
# ----------------------------------------------------------------------
def enumerate_meta() -> pd.DataFrame:
    """Load 4A's compiled (storm × admin × draw) metadata, project to the
    columns needed to construct 4B output paths."""
    meta = pd.read_parquet(COMPILED_4A_PARQUET, columns=ID_COLS)
    print(f"Loaded 4A inventory: {len(meta):,} expected 4B outputs.")
    return meta


# ----------------------------------------------------------------------
# Step 2 — chunked parallel read + concat
# ----------------------------------------------------------------------
def _read_chunk(meta_chunk: pd.DataFrame) -> pd.DataFrame:
    """Build paths for the chunk and read each parquet. Missing files are
    skipped silently (safe against partially-completed 4B); other read
    errors are logged and skipped — the consolidate step shouldn't crash
    on one corrupt parquet.

    Stamps each row with path-derived metadata that the 4B worker doesn't
    write into the output parquet itself:
      - source_id_variant_label: "{source_id}_{variant_label}"
      - experiment_id
      - basin
    """
    dfs = []
    for row in meta_chunk.itertuples(index=False):
        path = _build_4b_path(row)
        try:
            df = pd.read_parquet(path)
            df["source_id_variant_label"] = f"{row.source_id}_{row.variant_label}"
            df["experiment_id"] = row.experiment_id
            df["basin"] = row.basin
            df = df.drop(columns=["special_region_flag"], errors="ignore")
            dfs.append(df)
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"⚠️ {path}: {type(e).__name__}: {e}")
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _chunk_df(df: pd.DataFrame, chunk_size: int):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]


def _chunk_batches(df: pd.DataFrame, chunk_size: int, batch_chunks: int):
    batch = []
    for chunk in _chunk_df(df, chunk_size):
        batch.append(chunk)
        if len(batch) == batch_chunks:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_manifest_done() -> set[int]:
    done: set[int] = set()
    if not MANIFEST_JSONL.exists():
        return done
    with MANIFEST_JSONL.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "done" and "batch_idx" in rec:
                done.add(int(rec["batch_idx"]))
    return done


def _append_manifest(rec: dict):
    with MANIFEST_JSONL.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _write_part_file(dfs: list[pd.DataFrame], part_path: Path):
    tmp_path = part_path.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = None
    schema = None
    rows = 0
    try:
        for df in dfs:
            if df.empty:
                continue
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp_path, schema=schema, compression="snappy")
            else:
                if table.schema != schema:
                    table = table.cast(schema, safe=False)
            writer.write_table(table)
            rows += len(df)
    finally:
        if writer is not None:
            writer.close()

    if rows == 0:
        if tmp_path.exists():
            tmp_path.unlink()
        return 0

    tmp_path.replace(part_path)
    return rows


def _build_master_from_parts() -> int:
    parts = sorted(PARTS_DIR.glob("part_*.parquet"))
    if not parts:
        pd.DataFrame().to_parquet(CONSOLIDATED_PARQUET, index=False)
        return 0

    tmp_master = CONSOLIDATED_PARQUET.with_suffix(".parquet.tmp")
    if tmp_master.exists():
        tmp_master.unlink()

    writer = None
    schema = None
    total_rows = 0
    try:
        for part_path in parts:
            pf = pq.ParquetFile(part_path)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                # Rename `draw` → `tc_risk_draw` so the consolidated output
                # clearly identifies this as the inner CLIMADA draw (0-99),
                # not the outer storm_draw.
                if "draw" in table.schema.names:
                    table = table.rename_columns(
                        ["tc_risk_draw" if n == "draw" else n for n in table.schema.names]
                    )
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(tmp_master, schema=schema, compression="snappy")
                else:
                    if table.schema != schema:
                        table = table.cast(schema, safe=False)
                writer.write_table(table)
                total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if CONSOLIDATED_PARQUET.exists():
        CONSOLIDATED_PARQUET.unlink()
    tmp_master.replace(CONSOLIDATED_PARQUET)
    return total_rows


def compile_exposure(meta_df: pd.DataFrame) -> Path:
    total_chunks = (len(meta_df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_batches = (total_chunks + BATCH_CHUNKS - 1) // BATCH_CHUNKS
    print(
        f"Reading {len(meta_df):,} stage-4B parquets in {total_chunks:,} "
        f"chunks of {CHUNK_SIZE} across {NUM_CORES} cores "
        f"({total_batches:,} resumable batches)."
    )

    CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    done_batches = _load_manifest_done()
    total_rows_parts = 0
    for batch_idx, chunk_batch in enumerate(
        _chunk_batches(meta_df, CHUNK_SIZE, BATCH_CHUNKS),
        start=1,
    ):
        part_path = PARTS_DIR / f"part_{batch_idx:06d}.parquet"

        if batch_idx in done_batches and part_path.exists():
            print(f"Batch {batch_idx:,}/{total_batches:,}: already done, skipping.")
            continue

        start_ts = time.time()
        dfs = run_parallel(
            runner=_read_chunk,
            arg_list=chunk_batch,
            num_cores=NUM_CORES,
            progress_bar=False,
        )
        rows = _write_part_file(dfs=dfs, part_path=part_path)
        elapsed = time.time() - start_ts
        total_rows_parts += rows

        _append_manifest(
            {
                "batch_idx": batch_idx,
                "status": "done",
                "rows": rows,
                "seconds": round(elapsed, 3),
                "part_path": str(part_path),
                "timestamp": int(time.time()),
            }
        )
        print(
            f"Batch {batch_idx:,}/{total_batches:,}: wrote {rows:,} rows "
            f"in {elapsed:,.1f}s."
        )

    if WRITE_MASTER_PARQUET:
        total_rows_master = _build_master_from_parts()
    else:
        total_rows_master = -1

    print(
        f"Finished stage-4B consolidation. Parts rows written this run: "
        f"{total_rows_parts:,}."
    )
    if WRITE_MASTER_PARQUET:
        print(
            f"Saved consolidated 4B output → {CONSOLIDATED_PARQUET} "
            f"({total_rows_master:,} rows)"
        )
        return CONSOLIDATED_PARQUET

    print(f"Saved resumable part dataset under {PARTS_DIR}")
    return PARTS_DIR


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
if __name__ == "__main__":
    meta_df = enumerate_meta()
    compile_exposure(meta_df)
