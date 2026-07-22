import getpass
import uuid
from pathlib import Path

import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore


ADMIN_LEVEL = 0
YEARS = list(range(1970, 2026))  # 1970 to 2025
BASINS = ['EP', 'WP', 'SP', 'SI', 'NA', 'NI']

SAVE_ROOT = Path(
    f"/mnt/team/rapidresponse/pub/tropical-storms/climada/output/"
    f"ibtracs_stage4a_metadata_admin{ADMIN_LEVEL}"
)
SOURCE_ID = "ibtracs"
VARIANT_LABEL = "official"
EXPERIMENT_ID = "historical"


def _stage4a_yearly_metadata_path(year: int, basin: str) -> Path:
    """Mirror `save_yearly_exposure` in 04_admin_level_exposure_a_main.py."""
    save_dir = (
        SAVE_ROOT / SOURCE_ID / VARIANT_LABEL / EXPERIMENT_ID
        / str(year) / basin / "yearly_metadata"
    )
    filename = (
        f"metadata_{year}_{basin}_{SOURCE_ID}_{VARIANT_LABEL}_{EXPERIMENT_ID}.parquet"
    )
    return save_dir / filename


def task_is_complete(row) -> bool:
    """A stage-4a ibtracs task is complete iff its yearly metadata parquet
    exists and is at least 1 KB (the final artifact `save_yearly_exposure`
    writes after concatenating every per-storm parquet for the year)."""
    path = _stage4a_yearly_metadata_path(int(row.year), row.basin)
    return path.exists() and path.stat().st_size >= 1024


year_basin_combinations = pd.DataFrame(
    [(year, basin) for year in YEARS for basin in BASINS],
    columns=['year', 'basin'],
)

year_basin_combinations['runtime'] = '5m'
year_basin_combinations['memory'] = '15G'


############################################################################
# Completion derivation (filesystem-based, not Jobmon-based).
############################################################################
completed_mask = year_basin_combinations.apply(task_is_complete, axis=1)
remaining_meta = year_basin_combinations[~completed_mask].copy()
print(
    f"Completion scan: {completed_mask.sum()} / {len(year_basin_combinations)} "
    f"tasks already done on disk; {len(remaining_meta)} tasks to submit."
)


project = "proj_rapidresponse"
user = getpass.getuser()
wf_uuid = uuid.uuid4()

MAIN_SCRIPT = Path(__file__).resolve().parent / "04_admin_level_exposure_a_main.py"

tool = Tool(name="CLIMADA_stage4a_ibtracs")

workflow = tool.create_workflow(
    name=f"CLIMADA_stage4a_ibtracs_{wf_uuid}",
    # max_concurrently_running = 100,
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
        template_name=f"CLIMADA_stage4a_ibtracs_{config_key}",
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
            "--admin_level {admin_level}"
        ),
        node_args=["year", "basin", "admin_level"],
        task_args=[],
        op_args=[],
    )


tasks = []
for row in remaining_meta.itertuples():
    config_key = f"{row.runtime}_{row.memory}"
    template = task_templates[config_key]
    task = template.create_task(
        name=(
            f"CLIMADA_stage4a_ibtracs_"
            f"yr{row.year}_"
            f"{row.basin}_"
            f"admin{ADMIN_LEVEL}_"
            f"rt{row.runtime}_"
            f"mem{row.memory}"
        ),
        year=row.year,
        basin=row.basin,
        admin_level=ADMIN_LEVEL,
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
