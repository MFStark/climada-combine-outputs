import getpass
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
import os
import sys

DRAW_BATCHES = [
"0-4",
"5-9",
"10-14",
"15-19",
"20-24",
"25-29",
"30-34",
"35-39",
"40-44",
"45-49",
"50-54",
"55-59",
"60-64",
"65-69",
"70-74",
"75-79",
"80-84",
"85-89",
"90-94",
"95-99",
]

def assign_resources_single_core(row: pd.Series) -> pd.Series:
    n_admin0 = row["num_admin0_first_year"]
    n_years = row["num_years_in_batch"]

    # --- Runtime estimation ---
    slope_per_admin0 = (65.8 - 4.7) / (43 - 8)
    base_runtime_for_5yrs = 4.7 - slope_per_admin0 * 8

    runtime_min = (base_runtime_for_5yrs + slope_per_admin0 * n_admin0) * (n_years / 5)

    # enforce minimum runtime
    runtime_min = max(runtime_min, 4)

    # round to nearest 5 minutes
    runtime_rounded = int(round(runtime_min / 5) * 5)
    runtime_rounded = max(runtime_rounded, 5)

    # --- Memory estimation ---
    if n_admin0 <= 8:
        memory_gb = 20
    elif n_admin0 <= 21:
        memory_gb = 20 + (26 - 20) * (n_admin0 - 8) / (21 - 8)
    elif n_admin0 <= 43:
        memory_gb = 26 + (61 - 26) * (n_admin0 - 21) / (43 - 21)
    else:
        memory_gb = 61 + (n_admin0 - 43) * (61 - 26) / (43 - 21)

    # round to nearest 4 GB
    memory_rounded = int(round(memory_gb / 4) * 4)
    memory_rounded = max(memory_rounded, 4)

    row["memory_req"] = f"{memory_rounded}G"

    row["num_cores"] = 1
    row["max_run_time"] = runtime_rounded
    row["memory_req"] = f"{memory_rounded}G"

    return row


meta_df = pd.read_parquet("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/storm_draw_admin0_count.parquet")
meta_df = meta_df.drop(columns=["storm_draw", "direct_rr_draw", "indirect_cvd_draw", "indirect_resp_draw", "year"]).drop_duplicates().reset_index(drop=True)

meta_df = meta_df[
    (meta_df["source_id"] == "MPI-ESM1-2-HR") &
    (meta_df["experiment_id"] == "ssp126")
]

full_tasks = (meta_df
    .assign(key=1)
    .merge(pd.DataFrame({"draw_batch": DRAW_BATCHES, "key": 1}), on="key")
    .drop(columns=["key"])
)

# Assign run times based on storm counts
full_tasks_df = full_tasks.apply(assign_resources_single_core, axis=1)

user = getpass.getuser()

# Project
project = "proj_rapidresponse"  # Adjust this to your project name if needed

# create jobmon jobs
user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="CLIMADA_stage4")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"CLIMADA_stage4_{wf_uuid}",
    # max_concurrently_running = 100,
)


# Set resources on the workflow
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "5G",
        "cores": 2,
        "runtime": "5m",
        "constraints": "archive",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    }
)


# Get unique combinations of runtime, cores, and memory
unique_configs = full_tasks_df[['max_run_time', 'num_cores', 'memory_req']].drop_duplicates()

# Create task templates for each unique configuration
task_templates = {}
for _, config in unique_configs.iterrows():
    config_key = f"{config['max_run_time']}_{config['num_cores']}_{config['memory_req']}"
    
    task_templates[config_key] = tool.get_task_template(
        template_name=f"CLIMADA_stage4_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": config['num_cores'],
            "memory": config['memory_req'],
            "runtime": f"{int(config['max_run_time'])}m",
            "project": project,
        },
        command_template=(
            "python /ihme/homes/mfiking/github_repos/climada_python/script/climada/04_admin_level_exposure_main.py "
            "--source_id {source_id} "
            "--variant_label {variant_label} "
            "--experiment_id {experiment_id} "
            "--batch_year {batch_year} "
            "--basin {basin} "
            "--draw_batch {draw_batch}"
        ),
        node_args=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw_batch"],
        task_args=[],
        op_args=[],
    )


# Create tasks using the appropriate template
tasks = []
for row in full_tasks_df.itertuples():
    config_key = f"{row.max_run_time}_{row.num_cores}_{row.memory_req}"
    template = task_templates[config_key]

    task = template.create_task(
        name=(
            f"CLIMADA_stage4_"
            f"src{row.source_id}_"
            f"var{row.variant_label}_"
            f"exp{row.experiment_id}_"
            f"yr{row.batch_year}_"
            f"{row.basin}_"
            f"tracks_per_year{row.num_admin0_first_year}_"
            f"rt{row.max_run_time}m_"
            f"mem{row.memory_req}_"
            f"db{row.draw_batch}"
        ),
        source_id=row.source_id,
        variant_label=row.variant_label,
        experiment_id=row.experiment_id,
        batch_year=row.batch_year,
        basin=row.basin,
        draw_batch=row.draw_batch,
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
    print("✅ Workflow successfully bound.")
    print(f"Running workflow with ID {workflow.workflow_id}.")
    print("For full information see the Jobmon GUI:")
    print(f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")
except Exception as e:
    print(f"❌ Workflow binding failed: {e}")

try:
    status = workflow.run()
    print(f"Workflow {workflow.workflow_id} completed with status {status}.")
except Exception as e:
    print(f"❌ Workflow submission failed: {e}")
