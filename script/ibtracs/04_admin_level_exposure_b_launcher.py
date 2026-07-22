import re
import getpass
import uuid
from pathlib import Path

import pandas as pd  # type: ignore
from jobmon.client.tool import Tool  # type: ignore


ADMIN_LEVEL = 1

META_ROOT = Path(
    f"/mnt/team/rapidresponse/pub/tropical-storms/climada/output/"
    f"ibtracs_stage4a_metadata_admin{ADMIN_LEVEL}/ibtracs/official/historical"
)
SAVE_ROOT = Path(
    f"/mnt/team/rapidresponse/pub/tropical-storms/climada/output/"
    f"ibtracs_stage4b_pafs_admin{ADMIN_LEVEL}"
)
SOURCE_ID = "ibtracs"
VARIANT_LABEL = "official"
EXPERIMENT_ID = "historical"


def gather_completed_tasks(meta_df: pd.DataFrame) -> set:
    """
    Walk the stage 4B output tree once per (year, basin) combo and return a set
    of `(year, basin, storm_id)` tuples representing completed per-storm
    parquets. Files smaller than 1 KB are treated as incomplete.

    Stage 4B's save function writes each task's output as
    `storm_<storm_id>_<basin>_..._<year>.parquet` under
    `<year>/<basin>/storm_paf/`.
    """
    completed: set = set()
    filename_re = re.compile(r"^storm_(.+?)_")

    for combo in meta_df[["year", "basin"]].drop_duplicates().itertuples(index=False):
        storm_paf_dir = (
            SAVE_ROOT / SOURCE_ID / VARIANT_LABEL / EXPERIMENT_ID
            / str(int(combo.year)) / combo.basin / "storm_paf"
        )
        if not storm_paf_dir.exists():
            continue
        for parquet in storm_paf_dir.glob("storm_*.parquet"):
            if parquet.stat().st_size < 1024:
                continue
            m = filename_re.match(parquet.name)
            if not m:
                continue
            completed.add((int(combo.year), combo.basin, m.group(1)))
    return completed


# Inventory: union of every per-storm metadata parquet 4A wrote.
parquet_files = list(META_ROOT.glob("**/*.parquet"))
dfs = []
for pf in parquet_files:
    try:
        dfs.append(pd.read_parquet(pf))
    except Exception as e:
        print(f"Error reading {pf}: {e}")
full_df = pd.concat(dfs, ignore_index=True)
full_df = full_df[["storm_id", "year", "basin"]].drop_duplicates()

# Flat 5-min / 5G defaults — most historical storms are small.
full_df['runtime'] = '5m'
full_df['memory'] = '5G'


############################################################################
# Completion derivation (filesystem-based, not Jobmon-based).
#
# A stage-4B ibtracs task writes one per-storm parquet. We pre-walk the
# output tree once per (year, basin) combo, build a set of done
# (year, basin, storm_id) keys, then filter full_df with a single
# set-membership pass.
############################################################################
_completed_keys = gather_completed_tasks(full_df)
_meta_keys = list(zip(
    full_df["year"].astype(int),
    full_df["basin"],
    full_df["storm_id"].astype(str),
))
_completed_mask = pd.Series([k in _completed_keys for k in _meta_keys], index=full_df.index)
print(
    f"Completion scan: {_completed_mask.sum()} / {len(full_df)} tasks "
    f"already done on disk; {(~_completed_mask).sum()} tasks to submit."
)
full_df = full_df[~_completed_mask].copy()


project = "proj_rapidresponse"
user = getpass.getuser()
wf_uuid = uuid.uuid4()

MAIN_SCRIPT = Path(__file__).resolve().parent / "04_admin_level_exposure_b_main.py"

tool = Tool(name="CLIMADA_stage4b_ibtracs")

workflow = tool.create_workflow(
    name=f"CLIMADA_stage4b_ibtracs_{wf_uuid}",
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


unique_configs = full_df[['runtime', 'memory']].drop_duplicates()

task_templates = {}
for _, config in unique_configs.iterrows():
    config_key = f"{config['runtime']}_{config['memory']}"
    task_templates[config_key] = tool.get_task_template(
        template_name=f"CLIMADA_stage4b_ibtracs_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": 1,
            "memory": config['memory'],
            "runtime": f"{int(config['runtime'].replace('m', ''))}m",
            "project": project,
        },
        default_resource_scales={
            "memory": lambda x: int(x * 6),
            "runtime": lambda x: int(x * 1.25),
        },
        max_attempts=5,
        command_template=(
            f"python {MAIN_SCRIPT} "
            "--year {year} "
            "--basin {basin} "
            "--storm_id {storm_id} "
            "--admin_level {admin_level}"
        ),
        node_args=["year", "basin", "storm_id", "admin_level"],
        task_args=[],
        op_args=[],
    )


tasks = []
for row in full_df.itertuples():
    config_key = f"{row.runtime}_{row.memory}"
    template = task_templates[config_key]
    task = template.create_task(
        name=(
            f"CLIMADA_stage4b_ibtracs_"
            f"yr{row.year}_"
            f"{row.basin}_"
            f"st{row.storm_id}_"
            f"admin{ADMIN_LEVEL}_"
            f"rt{row.runtime}_"
            f"mem{row.memory}"
        ),
        year=row.year,
        basin=row.basin,
        storm_id=row.storm_id,
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
