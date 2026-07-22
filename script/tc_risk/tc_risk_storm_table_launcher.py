import getpass
from importlib.util import source_from_cache
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
import os
import sys

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
meta_df = meta_df[["source_id", "variant_label", "experiment_id"]].drop_duplicates().reset_index(drop=True)


#######################################################################
user = getpass.getuser()

# Project
project = "proj_rapidresponse"  # Adjust this to your project name if needed

# create jobmon jobs
user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="tc_risk_storm_table")

    
# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"tc_risk_storm_table_{wf_uuid}",
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


# Define the task template for processing each year batch
task_template = tool.get_task_template(
    template_name="tc_risk_storm_table_task",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": "all.q",
        "cores": 1,
        "memory": "20G",
        "runtime": "20m",
        "project": project,  # Ensure the project is set correctly
    },
    command_template=(
        "python /ihme/homes/mfiking/github_repos/climada_python/script/tc_risk/tc_risk_storm_table_main.py "
        "--source_id {source_id} "
        "--variant_label {variant_label} "
        "--experiment_id {experiment_id}"
    ),
    node_args=["source_id", "variant_label", "experiment_id"],  # 👈 Include model_dict in node_args
    task_args=[],  # Only variant is task-specific
    op_args=[],
)


tasks = []

for row in meta_df.itertuples():
    source_id = row.source_id
    variant_label = row.variant_label
    experiment_id = row.experiment_id
    task = task_template.create_task(
        name=f"tc_risk_storm_table_{source_id}_{variant_label}_{experiment_id}",
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
    )
    tasks.append(task)

print(f"Number of tasks: {len(tasks)}")


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
