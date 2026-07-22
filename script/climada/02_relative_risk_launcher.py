import uuid
import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path
import numpy as np

RELATIVE_RISKS = ["indirect_resp_draw", "indirect_cvd_draw"]

ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage2_v2/")


def _stage2_paf_path(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    relative_risk: str,
    sample_name: str,
    year: int,
) -> Path:
    """Path to a single year's raw_paf GeoTIFF. Must match the format
    written by `save_raster` in 02_relative_risk_main.py."""
    start_year, end_year = batch_year.split("-")
    return (
        SAVE_ROOT
        / storm_draw / source_id / variant_label / experiment_id
        / batch_year / str(year) / basin / "raw_paf"
        / (
            f"draw_mean_raw_paf_{storm_draw}_{relative_risk}_{sample_name}_"
            f"{basin}_{source_id}_{experiment_id}_{variant_label}_"
            f"{start_year}01_{end_year}12_{year}.tif"
        )
    )


def task_is_complete(row) -> bool:
    """A stage-2 task is complete iff every year in its batch_year has a
    raw_paf GeoTIFF on disk that's at least 1 KB. Mirrors the existence +
    size check in `check_if_draw_complete`; skips the load-validity check
    (the main script's resume logic will catch and rebuild corrupt files)."""
    start_year, end_year = map(int, row["batch_year"].split("-"))
    for year in range(start_year, end_year + 1):
        path = _stage2_paf_path(
            storm_draw=row["storm_draw"],
            source_id=row["source_id"],
            variant_label=row["variant_label"],
            experiment_id=row["experiment_id"],
            batch_year=row["batch_year"],
            basin=row["basin"],
            relative_risk=row["relative_risk"],
            sample_name=row["sample_name"],
            year=year,
        )
        if not path.exists() or path.stat().st_size < 1024:
            return False
    return True

# Read in paths
meta_df = pd.read_csv("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv")
meta_df = meta_df.drop(columns=["task_id", "draw"]).drop_duplicates()
# replace nan with NA
meta_df = meta_df.fillna("NA")

# Normalize column names
meta_df = meta_df.rename(columns={
    "model": "source_id",
    "variant": "variant_label",
    "scenario": "experiment_id",
    "time_period": "batch_year",
})

# Exclude the historical-baseline batch (1965-1969). Stage 2 runs on the
# projection batches only; the baseline outputs come from a separate workflow.
meta_df = meta_df[meta_df["batch_year"] != "1965-1969"]

# read in storm draws
storm_draw_df = pd.read_csv("/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv")

complete_df = meta_df.merge(
    storm_draw_df, 
    on=["source_id", "variant_label"],
    how="inner",
)

# replace storm_draw as storm_draw_XXXX
complete_df["storm_draw"] = complete_df["storm_draw"].apply(lambda x: f"storm_draw_{x:04d}")

# Ordered list of high-priority storm_draws. These are submitted first when
# PRIORITY_MODE == "priority"; submitted at the end of the queue when
# PRIORITY_MODE == "non_priority"; and shipped in priority order when
# PRIORITY_MODE == "all".
PRIORITY_DRAWS = [
    "storm_draw_0002",
    "storm_draw_0004",
    "storm_draw_0005",
    "storm_draw_0007",
    "storm_draw_0008",
    "storm_draw_0003",
    "storm_draw_0006",
    "storm_draw_0001",
]

# Which subset of storm_draws to submit. Change to flip the workflow.
#   "non_priority" — submit everything EXCEPT PRIORITY_DRAWS (default state)
#   "priority"     — submit ONLY PRIORITY_DRAWS (for testing / smoke runs)
#   "all"          — submit every storm_draw, priority ones first
PRIORITY_MODE = "all"

priority_map = {v: i for i, v in enumerate(PRIORITY_DRAWS)}

df = complete_df.copy()
df["draw_order"] = df["storm_draw"].map(priority_map).fillna(len(PRIORITY_DRAWS))
df = df.sort_values(
    ["draw_order", "storm_draw", "source_id", "variant_label",
     "experiment_id", "batch_year", "basin"]
).drop(columns="draw_order")

if PRIORITY_MODE == "non_priority":
    df = df[~df["storm_draw"].isin(PRIORITY_DRAWS)]
elif PRIORITY_MODE == "priority":
    df = df[df["storm_draw"].isin(PRIORITY_DRAWS)]
elif PRIORITY_MODE == "all":
    pass  # keep everything, priority-ordered
else:
    raise ValueError(
        f"Invalid PRIORITY_MODE={PRIORITY_MODE!r}; "
        f"expected one of: 'non_priority', 'priority', 'all'."
    )

#########################################################################################
# assign runtime
resource_df = pd.read_parquet("/mnt/share/homes/mfiking/downloads/climada_rs/stage2_resource_usage.parquet")
resource_df = resource_df.drop(columns=["task_id", "runtime", "memory", "memory_gb", "memory_rounded"])
resource_df = resource_df.rename(columns={
    "runtime_rounded": "max_run_time",
})

# assign 4GB
resource_df["memory_req"] = "4G"
# assign 10 cores
resource_df["num_cores"] = 10
resource_df = resource_df.drop_duplicates(subset=["source_id", "variant_label", "experiment_id", "batch_year", "basin"])

# merge with main df
final_df = df.merge(
    resource_df,
    on=["source_id", "variant_label", "experiment_id", "batch_year", "basin"],
    how="left"
)
# fill any na maxrun_time with 60 minutes, memory_req with 4G, num_cores with 10
final_df["max_run_time"] = final_df["max_run_time"].fillna(30)
final_df["memory_req"] = final_df["memory_req"].fillna("4G")
final_df["num_cores"] = final_df["num_cores"].fillna(10)


# Vectorized runtime bucketing (seconds): right-edge bins map to a small set
# of SLURM-friendly walltime ceilings. Values above 3600s fall into the 5700s
# bucket.
_RUNTIME_BIN_EDGES = [-float("inf"), 300, 600, 1200, 1800, 2700, 3600, float("inf")]
_RUNTIME_BIN_VALUES = [300, 600, 1200, 1800, 2700, 3600, 5700]

_bin_idx = pd.cut(
    final_df["max_run_time"],
    bins=_RUNTIME_BIN_EDGES,
    labels=False,
    include_lowest=True,
)
final_df["max_run_time"] = (
    pd.Series(_RUNTIME_BIN_VALUES)[_bin_idx].to_numpy()
)
# Convert from seconds to whole minutes (SLURM walltime granularity).
final_df["max_run_time"] = np.ceil(final_df["max_run_time"] / 60).astype(int)


#########################################################################################
# Fan out (indirect_resp_draw, indirect_cvd_draw) into separate rows.
rr_cols = ["indirect_cvd_draw", "indirect_resp_draw"]

final_long = final_df.melt(
    id_vars=[
        "source_id",
        "variant_label",
        "experiment_id",
        "batch_year",
        "basin",
        "storm_draw",
        "max_run_time",
        "runtime_min",
        "memory_req",
        "num_cores",
    ],
    value_vars=rr_cols,
    var_name="relative_risk",
    value_name="sample_name",
)

############################################################################################
# Completion derivation (filesystem-based, not Jobmon-based).
#
# A stage-2 task writes one raw_paf GeoTIFF per year in batch_year. A task is
# complete iff every year's TIF exists and is at least 1 KB. We mirror that
# check here per row of final_long and submit only rows whose 8-tuple isn't
# already fully written. The main script's check_if_draw_complete /
# get_year_status will handle the per-year recompute granularity inside any
# task we do submit.
completed_mask = final_long.apply(task_is_complete, axis=1)
remaining_long = final_long[~completed_mask].copy()
print(
    f"Completion scan: {completed_mask.sum()} / {len(final_long)} tasks "
    f"already done on disk; {len(remaining_long)} tasks to submit."
)

# Reruns: triple the budgeted runtime relative to the resource_usage parquet.
# (Empirical: invalidated single-year recomputes inside main() still re-run
# all 100 inner draws, so we want headroom over the first-time-success budget.)
remaining_long["max_run_time"] = remaining_long["max_run_time"] * 3


############################################################################################

project = "proj_rapidresponse"
wf_uuid = uuid.uuid4()

# Path to the worker script (resolved relative to this launcher so the repo
# can be moved without breaking the command).
MAIN_SCRIPT = Path(__file__).resolve().parent / "02_relative_risk_main.py"

tool = Tool(name="CLIMADA_stage2")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"CLIMADA_stage2_{wf_uuid}",
    # max_concurrently_running = 100,
)


# Set resources on the workflow
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "10G",
        "cores": 1,
        "runtime": "60m",
        "constraints": "archive",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    }
)


# Get unique combinations of runtime, cores, and memory
unique_configs = remaining_long[['max_run_time', 'num_cores', 'memory_req']].drop_duplicates()

# Create task templates for each unique configuration
task_templates = {}
for _, config in unique_configs.iterrows():
    config_key = f"{config['max_run_time']}_{config['num_cores']}_{config['memory_req']}"
    
    task_templates[config_key] = tool.get_task_template(
        template_name=f"CLIMADA_stage2_{config_key}",
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
            "--storm_draw {storm_draw} "
            "--source_id {source_id} "
            "--variant_label {variant_label} "
            "--experiment_id {experiment_id} "
            "--batch_year {batch_year} "
            "--basin {basin} "
            "--relative_risk {relative_risk} "
            "--sample_name {sample_name} "
            "--num_cores {num_cores}"
        ),
        node_args=["storm_draw", "source_id", "variant_label", "experiment_id", "batch_year", "basin", "relative_risk", "sample_name", "num_cores"],
        task_args=[],
        op_args=[],
    )


# Create tasks using the appropriate template
tasks = []
for row in remaining_long.itertuples():
    config_key = f"{row.max_run_time}_{row.num_cores}_{row.memory_req}"
    template = task_templates[config_key]

    task = template.create_task(
        name=(
            f"CLIMADA_stage2_"
            f"sd{row.storm_draw}_"
            f"src{row.source_id}_"
            f"var{row.variant_label}_"
            f"exp{row.experiment_id}_"
            f"yr{row.batch_year}_"
            f"{row.basin}_"
            f"{row.relative_risk}_"
            f"{row.sample_name}_"
            f"rt{row.max_run_time}m_"
            f"mem{row.memory_req}_"
            f"c{row.num_cores}"
        ),
        storm_draw=row.storm_draw,
        source_id=row.source_id,
        variant_label=row.variant_label,
        experiment_id=row.experiment_id,
        batch_year=row.batch_year,
        basin=row.basin,
        relative_risk=row.relative_risk,
        sample_name=row.sample_name,
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
