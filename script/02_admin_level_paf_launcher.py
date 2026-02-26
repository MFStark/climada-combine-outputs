import getpass
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
import os
import sys


SOURCE_IDS = ["ACCESS-CM2"]
VARIANT_LABELS = ["r1i1p1f1"]
EXPERIMENT_IDS = ["historical"]
BATCH_YEARS = ["1970-1989"]
YEARS = list(range(1970, 1990))
BASINS = ["EP"]
DRAWS = list(range(0, 1))  # Draws 0 to 99
RELATIVE_RISKS = ["rd", "cvd"]
SAMPLE_NAME = ["sample_099"]

user = getpass.getuser()

# Project
project = "proj_rapidresponse"  # Adjust this to your project name if needed

# create jobmon jobs
user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="RR_CLIMADA2")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"RR_CLIMADA2_{wf_uuid}",
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
    template_name="RR_CLIMADA2_task",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": "all.q",
        "cores": 1,
        "memory": "175G",
        "runtime": "30m",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    },
    command_template=(
        "python /ihme/homes/mfiking/github_repos/climada_python/script/climada/02_admin_level_paf_main.py "
        "--source_id {source_id} "
        "--variant_label {variant_label} "
        "--experiment_id {experiment_id} "
        "--batch_year {batch_year} "
        "--year {year} "
        "--basin {basin} "
        "--draw {draw} "
        "--relative_risk {relative_risk} "
        "--sample_name {sample_name}"
    ),
    node_args=["source_id", "variant_label", "experiment_id", "batch_year", "year", "basin", "draw", "relative_risk", "sample_name"],  # 👈 Include model_dict in node_args
    task_args=[],  # Only variant is task-specific
    op_args=[],
)

tasks = []

for source_id in SOURCE_IDS:
    for variant_label in VARIANT_LABELS:
        for experiment_id in EXPERIMENT_IDS:
            for batch_year in BATCH_YEARS:
                for year in YEARS:
                    for basin in BASINS:
                        for draw in DRAWS:
                            for relative_risk in RELATIVE_RISKS:
                                for sample_name in SAMPLE_NAME:
                                    task = task_template.create_task(
                                        name=f"RR_CLIMADA2_{source_id}_{variant_label}_{experiment_id}_{batch_year}_{year}_{basin}_d{draw}_{relative_risk}_{sample_name}",
                                        source_id=source_id,
                                        variant_label=variant_label,
                                        experiment_id=experiment_id,
                                        batch_year=batch_year,
                                        year=year,
                                        basin=basin,
                                        draw=draw,
                                        relative_risk=relative_risk,
                                        sample_name=sample_name,
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
