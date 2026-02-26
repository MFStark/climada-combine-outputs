import getpass
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
import os
import sys

# Manual TEST
SOURCE_IDS = ["CMCC-ESM2"]
VARIANT_LABELS = ["r1i1p1f1"]
EXPERIMENT_IDS = ["historical"]
BATCH_YEARS = ["1965-1974"]
BASINS = ["NI"]
DRAW_BATCHES = [
    "0-49",
    "50-99",
    "100-149",
    "150-199",
    "200-249",
]

# DRAW_BATCHES = [
#     "0-0",
# ]

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
    name=f"CLIMADA_stage3_{wf_uuid}",
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
    template_name="CLIMADA_stage3_task",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": "all.q",
        "cores": 10,
        "memory": "125G",
        "runtime": "180m",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    },
    command_template=(
        "python /ihme/homes/mfiking/github_repos/climada_python/script/climada/03_admin_level_exposure_main.py "
        "--source_id {source_id} "
        "--variant_label {variant_label} "
        "--experiment_id {experiment_id} "
        "--batch_year {batch_year} "
        "--basin {basin} "
        "--draw_batch {draw_batch}"
    ),
    node_args=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw_batch"],  # 👈 Include model_dict in node_args
    task_args=[],  # Only variant is task-specific
    op_args=[],
)

tasks = []

# TEST RUN
for source_id in SOURCE_IDS:
    for variant_label in VARIANT_LABELS:
        for experiment_id in EXPERIMENT_IDS:
            for batch_year in BATCH_YEARS:
                for basin in BASINS:
                    for draw_batch in DRAW_BATCHES:
                        task = task_template.create_task(
                            name=f"CLIMAD_stage3_{source_id}_{variant_label}_{experiment_id}_{batch_year}_{basin}_{draw_batch}",
                            source_id=source_id,
                            variant_label=variant_label,
                            experiment_id=experiment_id,
                            batch_year=batch_year,
                            basin=basin,
                            draw_batch=draw_batch,
                        )
                        tasks.append(task)   # ← must be here

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
