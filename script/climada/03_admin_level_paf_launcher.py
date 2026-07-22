import subprocess
import sys
import uuid
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path

RELATIVE_RISKS = ["indirect_resp_draw", "indirect_cvd_draw"]

STORM_DRAW_TABLE_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv")

# ── Admin level ───────────────────────────────────────────────────────────────
# Set to 0 or 1. Controls which metadata parquet, shapefile, and output root
# are used. All other launcher logic is identical across levels.
ADMIN_LEVEL = 1

if ADMIN_LEVEL == 0:
    SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage3_v2/")
    META_PARQUET = Path(
        "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/storm_draw_admin0_count.parquet"
    )
    GENERATE_SCRIPT = Path(__file__).resolve().parent / "generate_storm_draw_admin0_count.py"
    METRIC_COLS = [
        "n_storms_in_batch", "estimated_storms_per_year", "year",
        "num_admin0_first_year", "num_years_in_batch", "estimated_admin0_total",
    ]
    _ESTIMATED_TOTAL_COL = "estimated_admin0_total"
    # Derived from workflow 584066 (8 priority draws): observed max 88G / 45min.
    # Memory capped at 50G; ×2 retry gives 100G on attempt 2.
    _RESOURCE_TABLE = pd.DataFrame([
        {"resource_bin": "heavy", "memory_gb_assigned": 50, "runtime_min_assigned": 50},
        {"resource_bin": "light", "memory_gb_assigned": 50, "runtime_min_assigned": 35},
    ])
elif ADMIN_LEVEL == 1:
    SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage3_v2_admin1/")
    META_PARQUET = Path(
        "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/storm_draw_admin1_count.parquet"
    )
    GENERATE_SCRIPT = Path(__file__).resolve().parent / "generate_storm_draw_admin1_count.py"
    METRIC_COLS = [
        "n_storms_in_batch", "estimated_storms_per_year", "year",
        "num_admin1_first_year", "num_years_in_batch", "estimated_admin1_total",
    ]
    _ESTIMATED_TOTAL_COL = "estimated_admin1_total"
    # Admin1 resources from priority draw profiling.
    # heavy (0–100): max 3.61G / 7min  → 4G / 8min
    # light (100+):  max 9.80G / 4min  → 10G / 5min
    # default (estimated_admin1_total == 0, SP/SI tasks with storms in later
    # years but no year-1 admin1 intersection): 8G / 8min
    _RESOURCE_TABLE = pd.DataFrame([
        {"resource_bin": "heavy", "memory_gb_assigned": 4, "runtime_min_assigned": 8},
        {"resource_bin": "light", "memory_gb_assigned": 10, "runtime_min_assigned": 5},
    ])
    DEFAULT_MAX_RUN_TIME_MIN = 8
    DEFAULT_MEMORY_GB = 8
else:
    raise ValueError(f"ADMIN_LEVEL must be 0 or 1, got {ADMIN_LEVEL!r}")

UNIQUE_KEYS = ["source_id", "variant_label", "experiment_id", "batch_year", "basin"]

_RESOURCE_BINS   = [0, 100, np.inf]
_RESOURCE_LABELS = ["heavy", "light"]

if not META_PARQUET.exists():
    print(f"Metadata parquet not found at {META_PARQUET}. Running generation script...")
    subprocess.run([sys.executable, str(GENERATE_SCRIPT)], check=True)
    print("Generation complete.")


def _stage3_paf_path(
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
    """Path to a single year's per-admin PAF parquet. Must match the format
    written by `save_batch_paf_dataframe` in 03_admin_level_paf_main.py."""
    start_year, end_year = batch_year.split("-")
    return (
        SAVE_ROOT
        / storm_draw / source_id / variant_label / experiment_id
        / batch_year / basin / str(year) / "paf_df"
        / (
            f"paf_{storm_draw}_{relative_risk}_{sample_name}_{basin}_"
            f"{source_id}_{experiment_id}_{variant_label}_"
            f"{start_year}01_{end_year}12_{year}.parquet"
        )
    )


def task_is_complete(row) -> bool:
    """A stage-3 task is complete iff every year in its batch_year has a
    paf parquet on disk that's at least 1 KB. Mirrors the existence + size
    check in `check_if_year_complete`; skips the parquet header validity
    test (the main script's resume logic will catch and rebuild invalid
    files at runtime)."""
    start_year, end_year = map(int, row["batch_year"].split("-"))
    for year in range(start_year, end_year + 1):
        path = _stage3_paf_path(
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


# Resource constants applied uniformly to every task. (A piecewise-linear
# estimator scaled by admin0 count + batch length used to live here; it's
# preserved in git history. Reintroduce only if memory / runtime profiling
# justifies the complexity.)
DEFAULT_NUM_CORES = 1
DEFAULT_MAX_RUN_TIME_MIN = 10
DEFAULT_MEMORY_GB = 10


def assign_resources(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["num_cores"] = DEFAULT_NUM_CORES
    df["max_run_time"] = DEFAULT_MAX_RUN_TIME_MIN
    df["memory_gb"] = DEFAULT_MEMORY_GB
    df["resource_bin"] = pd.cut(
        df[_ESTIMATED_TOTAL_COL], bins=_RESOURCE_BINS, labels=_RESOURCE_LABELS, right=True
    ).astype(str)
    df = df.merge(_RESOURCE_TABLE, on="resource_bin", how="left")
    matched = df["memory_gb_assigned"].notna()
    df.loc[matched, "memory_gb"] = df.loc[matched, "memory_gb_assigned"].astype(int)
    df.loc[matched, "max_run_time"] = df.loc[matched, "runtime_min_assigned"].astype(int)
    df = df.drop(columns=["resource_bin", "memory_gb_assigned", "runtime_min_assigned"])
    print(f"Resources: {matched.sum()} tasks from table, {(~matched).sum()} using defaults.")
    print(
        f"Memory range: {df['memory_gb'].min()}G-{df['memory_gb'].max()}G | "
        f"Runtime: {df['max_run_time'].min()}-{df['max_run_time'].max()} min"
    )
    return df


meta_df = pd.read_parquet(META_PARQUET)

# The parquet contains metrics for 8 representative draws (one per unique
# source_id/variant_label). Propagate those metrics to all 100 storm draws
# by joining the full storm draw table on the 5-tuple keys.
_metrics_df = meta_df[UNIQUE_KEYS + METRIC_COLS].drop_duplicates(subset=UNIQUE_KEYS)
_storm_draw_df = pd.read_csv(STORM_DRAW_TABLE_PATH)
_storm_draw_df["storm_draw"] = _storm_draw_df["storm_draw"].apply(lambda x: f"storm_draw_{x:04d}")
_task_combos = meta_df[UNIQUE_KEYS].drop_duplicates()

full_tasks_df = (
    _storm_draw_df
    .merge(_task_combos, on=["source_id", "variant_label"], how="inner")
    .merge(_metrics_df, on=UNIQUE_KEYS, how="left")
)
print(f"Task list: {len(full_tasks_df)} rows across {full_tasks_df['storm_draw'].nunique()} storm draws.")

full_tasks_df = assign_resources(full_tasks_df)

# order tasks
combo_cols = ["source_id", "variant_label"]

full_tasks_df["draw_rank_within_model"] = (
    full_tasks_df
    .sort_values("storm_draw")
    .groupby(combo_cols)["storm_draw"]
    .rank(method="dense")
    .astype(int)
)
full_tasks_df = full_tasks_df.sort_values(
    [
        "draw_rank_within_model",  # round-robin across models
        "source_id",
        "variant_label",
        "storm_draw",
        "experiment_id",
        "batch_year",
        "basin",
    ]
).reset_index(drop=True)

##################
# High-priority storm_draws (kept in priority order for queue placement).
PRIORITY_DRAWS = [
    "storm_draw_0001",
    "storm_draw_0002",
    "storm_draw_0003",
    "storm_draw_0009",
    "storm_draw_0013",
    "storm_draw_0050",
    "storm_draw_0057",
    "storm_draw_0081",
]

# Which subset of storm_draws to submit:
#   "non_priority" — submit everything EXCEPT PRIORITY_DRAWS (default state)
#   "priority"     — submit ONLY PRIORITY_DRAWS (for testing / smoke runs)
#   "all"          — submit every storm_draw, priority ones first
PRIORITY_MODE = "all"

if PRIORITY_MODE == "non_priority":
    full_tasks_df = full_tasks_df[~full_tasks_df["storm_draw"].isin(PRIORITY_DRAWS)]
elif PRIORITY_MODE == "priority":
    full_tasks_df = full_tasks_df[full_tasks_df["storm_draw"].isin(PRIORITY_DRAWS)]
elif PRIORITY_MODE == "all":
    pass  # keep everything, priority-ordered by the upstream sort
else:
    raise ValueError(
        f"Invalid PRIORITY_MODE={PRIORITY_MODE!r}; "
        f"expected one of: 'non_priority', 'priority', 'all'."
    )

########################################################
# Fan out (indirect_resp_draw, indirect_cvd_draw) into separate rows.
rr_cols = ["indirect_cvd_draw", "indirect_resp_draw"]

final_long = full_tasks_df.melt(
    id_vars=[
        "source_id",
        "variant_label",
        "experiment_id",
        "batch_year",
        "basin",
        "storm_draw",
        "max_run_time",
        "memory_gb",
        "num_cores",
    ],
    value_vars=rr_cols,
    var_name="relative_risk",
    value_name="sample_name",
)

############################################################################################
# Completion derivation (filesystem-based, not Jobmon-based).
#
# A stage-3 task writes one paf parquet per year in batch_year. A task is
# complete iff every year's parquet exists and is at least 1 KB. Mirrors
# stage 2's launcher pattern. This also fixes a latent merge bug in the
# prior Jobmon-based completion: the old merge on= list omitted sample_name,
# so any one completed sample for a 7-tuple silently marked all other
# samples for the same 7-tuple as done. task_is_complete keys on the full
# 8-tuple, so each sample is now evaluated independently.
completed_mask = final_long.apply(task_is_complete, axis=1)
remaining_long = final_long[~completed_mask].copy()
print(
    f"Completion scan: {completed_mask.sum()} / {len(final_long)} tasks "
    f"already done on disk; {len(remaining_long)} tasks to submit."
)

# Format memory as a "{int}G" string for Jobmon's resource spec at the very
# last moment (kept as int up to this point for clean arithmetic).
remaining_long["memory_req"] = remaining_long["memory_gb"].astype(str) + "G"

###############################################################
project = "proj_rapidresponse"
wf_uuid = uuid.uuid4()

# Path to the worker script (resolved relative to this launcher so the repo
# can be moved without breaking the command).
MAIN_SCRIPT = Path(__file__).resolve().parent / "03_admin_level_paf_main.py"

tool = Tool(name=f"CLIMADA_stage3_admin{ADMIN_LEVEL}")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"CLIMADA_stage3_admin{ADMIN_LEVEL}_{wf_uuid}",
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
        template_name=f"CLIMADA_stage3_admin{ADMIN_LEVEL}_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": config['num_cores'],
            "memory": config['memory_req'],
            "runtime": f"{int(config['max_run_time'])}m",
            "project": project,
        },
        default_resource_scales={
            "memory": 1.25,
            "runtime": 1.75,
        },
        max_attempts=5,
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
            "--num_cores {num_cores} "
            f"--admin_level {ADMIN_LEVEL}"
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
            f"CLIMADA_stage3_a{ADMIN_LEVEL}_"
            f"sd{row.storm_draw}_"
            f"src{row.source_id}_"
            f"var{row.variant_label}_"
            f"exp{row.experiment_id}_"
            f"yr{row.batch_year}_"
            f"{row.basin}_"
            f"{row.relative_risk}_"
            f"s{row.sample_name}_"
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


###################################################################

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
