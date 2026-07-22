import getpass
import uuid
from pathlib import Path

import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore


YEARS = list(range(1970, 2026))  # 1970 to 2025
BASINS = ['EP', 'WP', 'SP', 'SI', 'NA', 'NI']

SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage1/")
SOURCE_ID = "ibtracs"
VARIANT_LABEL = "official"
EXPERIMENT_ID = "historical"


def _task_complete_marker_path(year: int, basin: str) -> Path:
    """Mirror `_task_complete_marker_path` in 01_climada_intensity_main.py."""
    return (
        SAVE_ROOT / SOURCE_ID / VARIANT_LABEL / EXPERIMENT_ID
        / str(year) / basin / "_task_complete.flag"
    )


def task_is_complete(row) -> bool:
    """A stage-1 ibtracs task is complete iff the main script wrote its
    `_task_complete.flag` sentinel after finishing the (year, basin)."""
    return _task_complete_marker_path(int(row.year), row.basin).exists()


# Build (year, basin) combinations
year_basin_combinations = pd.DataFrame(
    [(year, basin) for year in YEARS for basin in BASINS],
    columns=['year', 'basin'],
)


# Per-basin runtime budget (rough, basin-traffic-dependent). Memory is a flat
# 20G — the basins that genuinely need more get caught by the resource scale.
def assign_runtime(basin):
    if basin in ['NA', 'WP']:
        return '60m'
    elif basin == 'SP':
        return '20m'
    elif basin == 'SI':
        return '15m'
    elif basin in ['EP', 'NI']:
        return '25m'
    else:
        return '15m'


year_basin_combinations['runtime'] = year_basin_combinations['basin'].apply(assign_runtime)
year_basin_combinations['memory'] = '7G'


############################################################################
# Completion derivation (filesystem-based, not Jobmon-based).
#
# A stage-1 ibtracs task writes its `_task_complete.flag` sentinel at the
# end of main() — both for years with storms and for empty years (where
# the early "No storms found" path also touches the flag). Tasks whose
# flag is missing get submitted; otherwise we skip them.
############################################################################
completed_mask = year_basin_combinations.apply(task_is_complete, axis=1)
remaining_meta = year_basin_combinations[~completed_mask].copy()
print(
    f"Completion scan: {completed_mask.sum()} / {len(year_basin_combinations)} "
    f"tasks already done on disk; {len(remaining_meta)} tasks to submit."
)


user = getpass.getuser()
project = "proj_rapidresponse"
wf_uuid = uuid.uuid4()

# Path to the worker script (resolved relative to this launcher so the repo
# can be moved without breaking the command).
MAIN_SCRIPT = Path(__file__).resolve().parent / "01_climada_intensity_main.py"

# Create a tool
tool = Tool(name="IBTracs")


workflow = tool.create_workflow(
    name=f"IBTracs_{wf_uuid}",
    # max_concurrently_running = 100,
)


workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "5G",
        "cores": 1,
        "runtime": "5m",
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
        template_name=f"CLIMADA_stage1_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": 1,
            "memory": config['memory'],
            "runtime": f"{int(config['runtime'].replace('m', ''))}m",
            "project": project,
        },
        default_resource_scales={
            "memory": lambda x: int(x * 1.5),
            "runtime": lambda x: int(x * 1.5),
        },
        max_attempts=5,
        command_template=(
            f"python {MAIN_SCRIPT} "
            "--year {year} "
            "--basin {basin}"
        ),
        node_args=["year", "basin"],
    )


tasks = []
for row in remaining_meta.itertuples():
    config_key = f"{row.runtime}_{row.memory}"
    template = task_templates[config_key]
    task = template.create_task(
        name=(
            f"ibtracs_stage1_"
            f"year{row.year}_"
            f"basin{row.basin}_"
            f"rt{row.runtime}_"
            f"mem{row.memory}_"
        ),
        year=row.year,
        basin=row.basin,
    )
    tasks.append(task)


print(f"Number of tasks: {len(tasks)}")

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
