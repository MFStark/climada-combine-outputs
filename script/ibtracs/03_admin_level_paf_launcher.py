import getpass
import uuid
from pathlib import Path

import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore


YEARS = list(range(1970, 2026))  # 1970 to 2025
BASINS = ['EP', 'WP', 'SP', 'SI', 'NA', 'NI']
RELATIVE_RISKS = ["indirect_resp_draw", "indirect_cvd_draw"]

# ── Admin level ───────────────────────────────────────────────────────────────
# Set to 0 or 1. Controls output root and shapefile used by the main script.
ADMIN_LEVEL = 1

SOURCE_ID = "ibtracs"
VARIANT_LABEL = "official"
EXPERIMENT_ID = "historical"
SAMPLE_NAME = "mean"

if ADMIN_LEVEL == 0:
    SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage3")
elif ADMIN_LEVEL == 1:
    SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage3_admin1")
else:
    raise ValueError(f"ADMIN_LEVEL must be 0 or 1, got {ADMIN_LEVEL!r}")


def _stage3_paf_path(year: int, basin: str, relative_risk: str) -> Path:
    """Mirror `save_batch_paf_dataframe` in 03_admin_level_paf_main.py."""
    save_dir = (
        SAVE_ROOT / SOURCE_ID / VARIANT_LABEL / EXPERIMENT_ID
        / str(year) / basin / "paf_df"
    )
    filename = (
        f"paf_{relative_risk}_{SAMPLE_NAME}_{basin}_{SOURCE_ID}_"
        f"{EXPERIMENT_ID}_{VARIANT_LABEL}_{year}.parquet"
    )
    return save_dir / filename


def task_is_complete(row) -> bool:
    """A stage-3 ibtracs task is complete iff its per-year PAF parquet
    exists and is at least 1 KB (matches `check_if_year_complete` in the
    main script, minus the parquet-row validity check)."""
    path = _stage3_paf_path(int(row.year), row.basin, row.relative_risk)
    return path.exists() and path.stat().st_size >= 1024


year_basin_rr_combinations = pd.DataFrame(
    [(year, basin, rr) for year in YEARS for basin in BASINS for rr in RELATIVE_RISKS],
    columns=['year', 'basin', 'relative_risk'],
)

# Flat 3-min / 15G defaults (historical year, single basin)
year_basin_rr_combinations['runtime'] = '3m'
year_basin_rr_combinations['memory'] = '20G'


############################################################################
# Completion derivation (filesystem-based, not Jobmon-based).
############################################################################
completed_mask = year_basin_rr_combinations.apply(task_is_complete, axis=1)
remaining_meta = year_basin_rr_combinations[~completed_mask].copy()
print(
    f"Completion scan: {completed_mask.sum()} / {len(year_basin_rr_combinations)} "
    f"tasks already done on disk; {len(remaining_meta)} tasks to submit."
)


project = "proj_rapidresponse"
user = getpass.getuser()
wf_uuid = uuid.uuid4()

MAIN_SCRIPT = Path(__file__).resolve().parent / "03_admin_level_paf_main.py"

tool = Tool(name=f"CLIMADA_stage3_ibtracs_admin{ADMIN_LEVEL}")

workflow = tool.create_workflow(
    name=f"CLIMADA_stage3_ibtracs_admin{ADMIN_LEVEL}_{wf_uuid}",
)

workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "10G",
        "cores": 1,
        "runtime": "60m",
        "constraints": "archive",
        "queue": "all.q",
        "project": project,
    }
)


unique_configs = remaining_meta[['runtime', 'memory']].drop_duplicates()

task_templates = {}
for _, config in unique_configs.iterrows():
    config_key = f"{config['runtime']}_{config['memory']}"
    task_templates[config_key] = tool.get_task_template(
        template_name=f"CLIMADA_stage3_ibtracs_admin{ADMIN_LEVEL}_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": 1,
            "memory": config['memory'],
            "runtime": f"{int(config['runtime'].replace('m', ''))}m",
            "project": project,
        },
        command_template=(
            f"python {MAIN_SCRIPT} "
            "--year {year} "
            "--basin {basin} "
            "--relative_risk {relative_risk} "
            f"--admin_level {ADMIN_LEVEL}"
        ),
        node_args=["year", "basin", "relative_risk"],
        task_args=[],
        op_args=[],
    )


tasks = []
for row in remaining_meta.itertuples():
    config_key = f"{row.runtime}_{row.memory}"
    template = task_templates[config_key]
    task = template.create_task(
        name=(
            f"CLIMADA_stage3_ibtracs_a{ADMIN_LEVEL}_"
            f"yr{row.year}_"
            f"{row.basin}_"
            f"{row.relative_risk}_"
            f"rt{row.runtime}_"
            f"mem{row.memory}"
        ),
        year=row.year,
        basin=row.basin,
        relative_risk=row.relative_risk,
    )
    tasks.append(task)


print(f"Number of tasks: {len(tasks)}")
print(f"Number of task templates created: {len(task_templates)}")

if tasks:
    workflow.add_tasks(tasks)
    print("✅ Tasks successfully added to workflow.")
else:
    print("ℹ️ No tasks to submit (all complete on disk).")

try:
    workflow.bind()
    print("✅ Workflow successfully bound.")
    print(f"Running workflow with ID {workflow.workflow_id}.")
    print("For full information see the Jobmon GUI:")
    print(f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")
except Exception as e:
    print(f"❌ Workflow binding failed: {e}")
    raise SystemExit(1)

try:
    status = workflow.run()
    print(f"Workflow {workflow.workflow_id} completed with status {status}.")
except Exception as e:
    print(f"❌ Workflow submission failed: {e}")
