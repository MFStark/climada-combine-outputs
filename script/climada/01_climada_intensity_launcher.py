import uuid
import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path
import xarray as xr  # type: ignore
from rra_tools.parallel import run_parallel  # type: ignore

DRAWS_TOTAL = 100
DRAWS_PER_BATCH = 5
DRAW_BATCHES = [
    f"{i}-{i + DRAWS_PER_BATCH - 1}"
    for i in range(0, DRAWS_TOTAL, DRAWS_PER_BATCH)
]

ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")


def read_custom_tracks_nc(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int = 0,
) -> xr.Dataset:

    start_year, end_year = batch_year.split("-")
    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    nc_file = (
        ROOT_PATH
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"tracks_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.nc"
    )

    if not nc_file.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_file}")

    return xr.open_dataset(nc_file)

def count_tracks(ds: xr.Dataset) -> int:
    return ds.sizes["n_trk"]

def single_storm_count(row: pd.Series) -> int:
    ds = read_custom_tracks_nc(
        source_id=row["source_id"],
        variant_label=row["variant_label"],
        experiment_id=row["experiment_id"],
        batch_year=row["batch_year"],
        basin=row["basin"],
    )
    num_tracks = count_tracks(ds)
    ds.close()
    return num_tracks

def run_storm_count_parallel(task_df: pd.DataFrame) -> pd.DataFrame:
    task_df = task_df.copy()
    task_df = task_df[["source_id", "variant_label", "experiment_id", "batch_year", "basin"]].drop_duplicates().reset_index(drop=True)
    task_df["num_tracks"] = run_parallel(
        runner=single_storm_count,
        arg_list=[row for _, row in task_df.iterrows()],
        num_cores=10,
        progress_bar=True,
    )
    return task_df[["source_id", "variant_label", "experiment_id", "batch_year", "basin", "num_tracks"]]

def assign_run_resources(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized resource assignment.
    Fixed per-task: cores=5, memory=35G.
    Runtime is binned from empirical performance + a 10-minute safety buffer.
    """
    df = df.copy()
    df["num_cores"] = 5
    df["memory_req"] = "35G"

    # Right-open bins: (-inf, 15], (15, 30], (30, 70], (70, 120], (120, 180], (180, inf)
    bin_edges = [-float("inf"), 15, 30, 70, 120, 180, float("inf")]
    base_runtimes = [40, 50, 80, 120, 190, 250]
    bin_idx = pd.cut(df["num_tracks"], bins=bin_edges, labels=False, include_lowest=True)
    df["max_run_time"] = pd.Series(base_runtimes)[bin_idx].to_numpy() + 10
    return df


# Read in paths
meta_df = pd.read_csv("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv")
meta_df = meta_df.drop(columns=["task_id", "draw"]).drop_duplicates()


# replace nan basin with "NA"
meta_df["basin"] = meta_df["basin"].fillna("NA")

# Normalize column names
meta_df = meta_df.rename(columns={
    "model": "source_id",
    "variant": "variant_label",
    "scenario": "experiment_id",
    "time_period": "batch_year",
})

# get storm counts
meta_df_storm_counts = run_storm_count_parallel(meta_df)

# Assign run times based on storm counts
meta_df_storm_counts = assign_run_resources(meta_df_storm_counts)

# # Create full tasks by cross-joining with draw batches
full_tasks_df = (
    meta_df_storm_counts[["source_id", "variant_label", "experiment_id", "batch_year", "basin", "num_tracks", "max_run_time", "num_cores", "memory_req"]]
    .assign(key=1)
    .merge(pd.DataFrame({"draw_batch": DRAW_BATCHES, "key": 1}), on="key")
    .drop(columns=["key"])
)

######################################################################
# Completion derivation (filesystem-based, not Jobmon-based).
#
# Stage 1's main script writes a per-draw JSON marker at
#   <LOG_DIR>/draw_completion_markers/<source>/<variant>/<exp>/<batch_year>/<basin>/draw_NNNN.json
# once a draw finishes. We treat the presence of every marker in a
# draw_batch as the batch being done.
#
# Granularity note (see STAGES.md A2): if a batch crashed mid-way (e.g. 3/5
# draws marked), the launcher reruns the WHOLE batch. The main script's
# per-storm resume logic short-circuits the already-done draws inside it,
# so cost is minimal — we just pay one task slot for the catch-up draws.
######################################################################
LOG_DIR = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1_v2_log/")
MARKER_ROOT = LOG_DIR / "draw_completion_markers"

completed_draws: set[tuple] = set()
if MARKER_ROOT.exists():
    for marker in MARKER_ROOT.rglob("draw_*.json"):
        parts = marker.relative_to(MARKER_ROOT).parts
        # expected: (source_id, variant_label, experiment_id, batch_year, basin, "draw_NNNN.json")
        if len(parts) != 6:
            continue
        try:
            draw = int(parts[5].removeprefix("draw_").removesuffix(".json"))
        except ValueError:
            continue
        completed_draws.add((parts[0], parts[1], parts[2], parts[3], parts[4], draw))


def _batch_is_complete(row: pd.Series) -> bool:
    """A draw_batch is complete iff every draw in it has a marker on disk."""
    start, end = map(int, row["draw_batch"].split("-"))
    key_prefix = (
        row["source_id"], row["variant_label"], row["experiment_id"],
        row["batch_year"], row["basin"],
    )
    return all((*key_prefix, d) in completed_draws for d in range(start, end + 1))


_batch_done_mask = full_tasks_df.apply(_batch_is_complete, axis=1)
remaining_meta = full_tasks_df[~_batch_done_mask].reset_index(drop=True)

print(
    f"Completion scan: {_batch_done_mask.sum()} / {len(full_tasks_df)} batches "
    f"already done on disk; {len(remaining_meta)} batches to submit."
)
# ----------------------------------------------------------------------
# Submission ordering
#
# Finish one (source_id, variant_label) combination fully before starting the
# next. MRI-ESM2-0 carries many variant_labels relative to other sources, so
# push it to the bottom so it doesn't dominate the front of the queue.
# ----------------------------------------------------------------------
DEPRIORITIZED_SOURCES = {"MRI-ESM2-0"}

new_meta = remaining_meta.copy()
new_meta["_deprio"] = new_meta["source_id"].isin(DEPRIORITIZED_SOURCES).astype(int)
new_meta = (
    new_meta
    .sort_values(
        [
            "_deprio",
            "source_id",
            "variant_label",
            "experiment_id",
            "batch_year",
            "basin",
            "draw_batch",
        ]
    )
    .drop(columns=["_deprio"])
    .reset_index(drop=True)
)

project = "proj_rapidresponse"  # Adjust this to your project name if needed
wf_uuid = uuid.uuid4()

# Path to the worker script (resolved relative to this launcher so the repo
# can be moved without breaking the command).
MAIN_SCRIPT = Path(__file__).resolve().parent / "01_climada_intensity_main.py"

# Create a tool
tool = Tool(name="CLIMADA_state1")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"CLIMADA_state1_{wf_uuid}",
    # max_concurrently_running = 100,
)


# Set resources on the workflow
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "5G",
        "cores": 1,
        "runtime": "5m",
        "constraints": "archive",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    }
)

# Get unique combinations of runtime, cores, and memory
unique_configs = new_meta[['max_run_time', 'num_cores', 'memory_req']].drop_duplicates()

# Create task templates for each unique configuration
task_templates = {}
for _, config in unique_configs.iterrows():
    config_key = f"rt{config['max_run_time']}_c{config['num_cores']}_m{config['memory_req']}"
    
    task_templates[config_key] = tool.get_task_template(
        template_name=f"CLIMADA_state1_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": config['num_cores'],
            "memory": config['memory_req'],
            "runtime": f"{int(config['max_run_time'])}m",
            "project": project,
        },
        command_template=(
            f"python {MAIN_SCRIPT} "
            "--source_id {source_id} "
            "--variant_label {variant_label} "
            "--experiment_id {experiment_id} "
            "--batch_year {batch_year} "
            "--basin {basin} "
            "--draw_batch {draw_batch} "
            "--num_cores {num_cores} "
        ),
        node_args=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw_batch", "num_cores"],
        task_args=[],
        op_args=[],
    )

# Create tasks using the appropriate template
tasks = []
for row in new_meta.itertuples():
    config_key = f"rt{row.max_run_time}_c{row.num_cores}_m{row.memory_req}"
    template = task_templates[config_key]
    
    task = template.create_task(
        name=f"CLIMADA_state1_{row.source_id}_{row.variant_label}_{row.experiment_id}_{row.batch_year}_{row.basin}_d{row.draw_batch}_c{row.num_cores}",
        source_id=row.source_id,
        variant_label=row.variant_label,
        experiment_id=row.experiment_id,
        batch_year=row.batch_year,
        basin=row.basin,
        draw_batch=row.draw_batch,
        num_cores=row.num_cores,
    )
    tasks.append(task)

print(f"Number of tasks: {len(tasks)}")
print(f"Number of task templates created: {len(task_templates)}")



if tasks:
    workflow.add_tasks(tasks)
    print("✅ Tasks successfully added to workflow.")
else:
    print("⚠️ No tasks added to workflow. Check task generation.")

try:
    workflow.bind()
except Exception as e:
    print(f"❌ Workflow binding failed: {e}")
    raise SystemExit(1)

print("✅ Workflow successfully bound.")
print(f"Running workflow with ID {workflow.workflow_id}.")
print("For full information see the Jobmon GUI:")
print(f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")

try:
    status = workflow.run()
    print(f"Workflow {workflow.workflow_id} completed with status {status}.")
except Exception as e:
    print(f"❌ Workflow submission failed: {e}")
