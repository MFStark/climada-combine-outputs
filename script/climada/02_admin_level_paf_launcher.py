import getpass
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
import os
import sys


SOURCE_IDS = ["CMCC-ESM2"]
VARIANT_LABELS = ["r1i1p1f1"]
EXPERIMENT_IDS = ["historical"]
BATCH_YEARS = ["1965-1974"]
BASINS = ["NI"]
RELATIVE_RISKS = ["rd", "cvd"]
SAMPLE_NAME = ["sample_099"]

# FULL Run
# SOURCE_IDS = ["CMCC-ESM2", "EC-Earth3", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
# RELATIVE_RISKS = ["rd", "cvd"]
# BASINS = ["EP", "NA", "NI", "SI", "AU", "SP", "WP"]

# registry_path = Path(
#     "/mnt/team/rapidresponse/pub/tropical-storms/tc_risk/input/cmip6/folder_paths_registry.csv"
# )
# reg_df = pd.read_csv(registry_path)

# reg_df = reg_df[["model", "variant", "scenario", "time_period"]].rename(
#     columns={
#         "model": "source_id",
#         "variant": "variant_label",
#         "scenario": "experiment_id",
#         "time_period": "batch_year",
#     }
# )

# reg_df = reg_df[reg_df["source_id"].isin(SOURCE_IDS)]

# storm_draw_df = pd.read_csv("/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv")

# complete_df = reg_df.merge(
#     storm_draw_df,
#     on=["source_id", "variant_label",],
#     how="inner",
# )

# # Number of basins
# n_basins = len(BASINS)

# # Repeat each row of complete_df for each basin
# expanded_df = complete_df.loc[complete_df.index.repeat(n_basins)].copy()

# # Assign basin values using np.tile
# expanded_df["basin"] = np.tile(BASINS, len(complete_df))

# # Reset index (optional)
# expanded_df = expanded_df.reset_index(drop=True)


user = getpass.getuser()

# Project
project = "proj_rapidresponse"  # Adjust this to your project name if needed

# create jobmon jobs
user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="CLIMADA_stage2")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"CLIMADA_stage2_{wf_uuid}",
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
    template_name="CLIMADA_stage2_task",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": "all.q",
        "cores": 8,
        "memory": "200G",
        "runtime": "600m",
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
        "--relative_risk {relative_risk} "
        "--sample_name {sample_name}"
    ),
    node_args=["source_id", "variant_label", "experiment_id", "batch_year", "year", "basin", "relative_risk", "sample_name"],  # 👈 Include model_dict in node_args
    task_args=[],  # Only variant is task-specific
    op_args=[],
)

tasks = []

for source_id in SOURCE_IDS:
    for variant_label in VARIANT_LABELS:
        for experiment_id in EXPERIMENT_IDS:
            for batch_year in BATCH_YEARS:
                start_year, end_year = map(int, batch_year.split("-"))
                for year in range(start_year, end_year + 1):
                    for basin in BASINS:
                        for relative_risk in RELATIVE_RISKS:
                            for sample_name in SAMPLE_NAME:
                                task = task_template.create_task(
                                    name=f"CLIMADA_stage2_{source_id}_{variant_label}_{experiment_id}_{batch_year}_{year}_{basin}_{relative_risk}_{sample_name}",
                                    source_id=source_id,
                                    variant_label=variant_label,
                                    experiment_id=experiment_id,
                                    batch_year=batch_year,
                                    year=year,
                                    basin=basin,
                                    relative_risk=relative_risk,
                                    sample_name=sample_name,
                                )
                                tasks.append(task)

# Full Run
# for row in expanded_df.itertuples():
#     batch_year = row.batch_year
#     start_year, end_year = map(int, batch_year.split("-"))
#     for year in range(start_year, end_year + 1):
#         for relative_risk in RELATIVE_RISKS:
#             if relative_risk == "rd":
#                 sample_name = row.indirect_resp_draw
#             elif relative_risk == "cvd":
#                 sample_name = row.indirect_cvd_draw
#             else:
#                 raise ValueError(f"Unexpected relative risk type: {relative_risk}")
#             task = task_template.create_task(
#                 name=f"CLIMADA_stage2_{row.storm_draw}_{row.source_id}_{row.variant_label}_{row.experiment_id}_{batch_year}_{year}_{row.basin}_{relative_risk}_{sample_name}",
#                 source_id=row.source_id,
#                 variant_label=row.variant_label,
#                 experiment_id=row.experiment_id,
#                 batch_year=batch_year,
#                 year=year,
#                 basin=row.basin,
#                 relative_risk=relative_risk,
#                 sample_name=sample_name,
#             )
#             tasks.append(task)

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
