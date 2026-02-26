import getpass
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
import os
import sys

YEARS = list(range(1980, 2026)) # 1980 t0 2025
BASINS = ['EP', 'WP', 'SP', 'SI', 'NA', 'NI']

user = getpass.getuser()

# Project
project = "proj_rapidresponse"  # Adjust this to your project name if needed

# create jobmon jobs
user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="IBTracks")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"IBTracks_{wf_uuid}",
    # max_concurrently_running = 100,
)


# Set resources on the workflow
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "5G",
        "cores": 1,
        "runtime": "5m",
        "constraints": "archive",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    }
)


# Define the task template for processing each year batch
task_template = tool.get_task_template(
    template_name="IBTracks_task",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": "all.q",
        "cores": 1,
        "memory": "150G",
        "runtime": "15m",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    },
    command_template=(
        "python /ihme/homes/mfiking/github_repos/climada_python/script/ibtracks/00_historical_main.py "
        "--year {year} "
        "--basin {basin}"
    ),
    node_args=["year", "basin"],  # 👈 Include model_dict in node_args
    task_args=[],  # Only variant is task-specific
    op_args=[],
)

tasks = []

for year in YEARS:
    for basin in BASINS:
        task = task_template.create_task(
            name=f"IBTracks_{year}_{basin}",
            year=year,
            basin=basin,
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
