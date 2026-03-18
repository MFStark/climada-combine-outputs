import getpass
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
import os
import sys
from rra_tools.parallel import run_parallel # type: ignore
import math 
import xarray as xr # type: ignore

RELATIVE_RISKS = ["indirect_resp_draw", "indirect_cvd_draw"]


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


# Assign run times based on storm counts
full_tasks_df = meta_df.apply(assign_resources_single_core, axis=1)

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
        "memory_req",
        "max_run_time",
    ]
).reset_index(drop=True)



user = getpass.getuser()

# Project
project = "proj_rapidresponse"  # Adjust this to your project name if needed

# create jobmon jobs
user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="CLIMADA_stage3")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"CLIMADA_stage3{wf_uuid}",
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
unique_configs = full_tasks_df[['max_run_time', 'num_cores', 'memory_req']].drop_duplicates()

# Create task templates for each unique configuration
task_templates = {}
for _, config in unique_configs.iterrows():
    config_key = f"{config['max_run_time']}_{config['num_cores']}_{config['memory_req']}"
    
    task_templates[config_key] = tool.get_task_template(
        template_name=f"CLIMADA_stage3_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": config['num_cores'],
            "memory": config['memory_req'],
            "runtime": f"{int(config['max_run_time'])}m",
            "project": project,
        },
        command_template=(
            "python /ihme/homes/mfiking/github_repos/climada_python/script/climada/03_admin_level_paf_main.py "
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
for row in full_tasks_df.itertuples():
    config_key = f"{row.max_run_time}_{row.num_cores}_{row.memory_req}"
    template = task_templates[config_key]

    for relative_risk in RELATIVE_RISKS:
        if relative_risk == "indirect_resp_draw":
            sample_name = row.indirect_resp_draw
        elif relative_risk == "indirect_cvd_draw":
            sample_name = row.indirect_cvd_draw
        else:
            raise ValueError(f"Unexpected relative risk type: {relative_risk}")

        task = template.create_task(
            name=(
                f"CLIMADA_stage3_"
                f"sd{row.storm_draw}_"
                f"src{row.source_id}_"
                f"var{row.variant_label}_"
                f"exp{row.experiment_id}_"
                f"yr{row.batch_year}_"
                f"{row.basin}_"
                f"{relative_risk}_"
                f"{sample_name}_"
                f"tracks_per_year{row.num_admin0_first_year}_"
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
            relative_risk=relative_risk,
            sample_name=sample_name,
            num_cores=row.num_cores,
        )

        tasks.append(task)

print(f"Number of tasks: {len(tasks)}")
print(f"Number of task templates created: {len(task_templates)}")



##################################################
# Subset completed tasks from original workflow  #
##################################################
workflow_id = 552899
df = workflow_tasks(
    workflow_id=workflow_id,
    limit=-1   # return all tasks
)
completed_tasks_list = df[df["STATUS"] == "DONE"]["TASK_NAME"].to_list()
# Convert completed task list to a set for faster lookup
completed_set = set(completed_tasks_list)


workflow_id2 = 552932
df2 = workflow_tasks(
    workflow_id=workflow_id2,
    limit=-1   # return all tasks
)
completed_tasks_list2 = df2[df2["STATUS"] == "DONE"]["TASK_NAME"].to_list()
# Convert completed task list to a set for faster lookup
completed_set2 = set(completed_tasks_list2)

# combine both setes
completed_set_union = completed_set.union(completed_set2)

# Filter tasks
tasks_to_run = [task for task in tasks if task.name not in completed_set_union]

print(f"Original tasks: {len(tasks)}")
print(f"Complted Tasks: {len(completed_tasks_list)}")
print(f"Tasks to rerun: {len(tasks_to_run)}")

###################################################################

if tasks_to_run:
    workflow.add_tasks(tasks_to_run)
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
