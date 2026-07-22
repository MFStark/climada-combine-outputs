import getpass
import uuid
from pathlib import Path

import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore


YEARS = list(range(1970, 2026))  # 1970 to 2025
BASINS = ['EP', 'WP', 'SP', 'SI', 'NA', 'NI']
RELATIVE_RISKS = ["indirect_resp_draw", "indirect_cvd_draw"]

SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage2")
SOURCE_ID = "ibtracs"
VARIANT_LABEL = "official"
EXPERIMENT_ID = "historical"


def _stage2_paf_path(year: int, basin: str, relative_risk: str) -> Path:
    """Mirror `save_raster` / `check_if_draw_complete` in 02_relative_risk_main.py."""
    save_dir = (
        SAVE_ROOT / SOURCE_ID / VARIANT_LABEL / EXPERIMENT_ID
        / str(year) / basin / "raw_paf"
    )
    filename = (
        f"draw_mean_raw_paf_{relative_risk}_{basin}_{SOURCE_ID}_"
        f"{EXPERIMENT_ID}_{VARIANT_LABEL}_{year}.tif"
    )
    return save_dir / filename


def task_is_complete(row) -> bool:
    """A stage-2 ibtracs task is complete iff its draw-mean PAF TIF exists
    and is at least 1 KB (mirrors `check_if_draw_complete` in the main
    script, minus the rasterra load which is too slow for a per-task
    launcher check)."""
    path = _stage2_paf_path(int(row.year), row.basin, row.relative_risk)
    return path.exists() and path.stat().st_size >= 1024


year_basin_rr_combinations = pd.DataFrame(
    [(year, basin, rr) for year in YEARS for basin in BASINS for rr in RELATIVE_RISKS],
    columns=['year', 'basin', 'relative_risk'],
)

# Flat 5-min / 5G defaults — historical basins are small enough that the
# scaled runtime/memory rarely fires (resource_scales below).
year_basin_rr_combinations['runtime'] = '1m'
year_basin_rr_combinations['memory'] = '1G'


############################################################################
# Completion derivation (filesystem-based, not Jobmon-based).
#
# A stage-2 ibtracs task writes one draw-mean raw_paf TIF per
# (year, basin, relative_risk). Tasks whose TIF is missing or < 1KB get
# submitted; otherwise we skip them.
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

MAIN_SCRIPT = Path(__file__).resolve().parent / "02_relative_risk_main.py"

tool = Tool(name="CLIMADA_stage2_ibtracs")

workflow = tool.create_workflow(
    name=f"CLIMADA_stage2_ibtracs_{wf_uuid}",
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
        template_name=f"CLIMADA_stage2_ibtracs_{config_key}",
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
            "--relative_risk {relative_risk}"
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
            f"CLIMADA_stage2_ibtracs_"
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
