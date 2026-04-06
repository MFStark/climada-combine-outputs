import getpass
import re
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
    # n_admin0 = row["num_admin0_first_year"]
    # n_years = row["num_years_in_batch"]

    # # --- Runtime estimation ---
    # slope_per_admin0 = (65.8 - 4.7) / (43 - 8)
    # base_runtime_for_5yrs = 4.7 - slope_per_admin0 * 8

    # runtime_min = (base_runtime_for_5yrs + slope_per_admin0 * n_admin0) * (n_years / 5)

    # # enforce minimum runtime
    # runtime_min = max(runtime_min, 4)

    # # round to nearest 5 minutes
    # runtime_rounded = int(round(runtime_min / 5) * 5)
    # runtime_rounded = max(runtime_rounded, 5)

    # # --- Memory estimation ---
    # if n_admin0 <= 8:
    #     memory_gb = 20
    # elif n_admin0 <= 21:
    #     memory_gb = 20 + (26 - 20) * (n_admin0 - 8) / (21 - 8)
    # elif n_admin0 <= 43:
    #     memory_gb = 26 + (61 - 26) * (n_admin0 - 21) / (43 - 21)
    # else:
    #     memory_gb = 61 + (n_admin0 - 43) * (61 - 26) / (43 - 21)

    # # round to nearest 4 GB
    # memory_rounded = int(round(memory_gb / 4) * 4)
    # memory_rounded = max(memory_rounded, 4)

    # # round runtime to nearest 5 minutes
    # runtime_rounded = int(round(runtime_rounded / 5) * 5)
    # runtime_rounded = max(runtime_rounded, 5)

    # basin = row["basin"]

    # # if basin in AU or EP
    # if basin in ["AU", "EP"]:
    #     memory_rounded = memory_rounded * 2
    #     runtime_rounded = runtime_rounded * 2

    # row["num_cores"] = 1
    # row["max_run_time"] = runtime_rounded
    # row["memory_req"] = f"{memory_rounded}G"

    # # bin memory into 20, 28, 36, 44, 52, 60, 68
    # memory_bins = [20, 28, 36, 44, 52, 60, 68]
    # memory_rounded = min(memory_bins, key=lambda x: abs(x - memory_rounded))
    # row["memory_req"] = f"{memory_rounded}G"

    # # bin runtime into 5, 10, 30, 60
    # runtime_bins = [5, 10, 30, 60]
    # runtime_rounded = min(runtime_bins, key=lambda x: abs(x - runtime_rounded))
    # row["max_run_time"] = runtime_rounded

    # for testing baseline
    row["num_cores"] = 1
    row["max_run_time"] = 10
    row["memory_req"] = "25G"
        
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

##################
priority = [
    "storm_draw_0002",
    "storm_draw_0004",
    "storm_draw_0005",
    "storm_draw_0007",
    "storm_draw_0008",
    "storm_draw_0003",
    "storm_draw_0006",
    "storm_draw_0001",
]

# subset to first 8 storm draws for testing
full_tasks_df = full_tasks_df[~full_tasks_df["storm_draw"].isin(priority)]

########################################################
# workflow_id1 = 558358 # priority storm draws
# workflow_id2 = 558359 # priority storm draws
# workflow_id3 = 558361 # priority storm draws

workflow_id1 = 558799
workflow_id2 = 559012
workflow_id3 = 559296
workflow_id4 = 559449
workflow_id5 = 559686
workflow_id6 = 559817
workflow_ids = [workflow_id1, workflow_id2, workflow_id3, workflow_id4, workflow_id5, workflow_id6]

total_df_list = []

for workflow_id in workflow_ids:
    df = workflow_tasks(
        workflow_id=workflow_id,
        limit=-1   # return all tasks
    )
    completed_df = df[df["STATUS"] == "D"]
    total_df_list.append(completed_df)


total_df = pd.concat(total_df_list, ignore_index=True)

# Create completed parameters df
parts = total_df["TASK_NAME"].str.split("_", expand=True)

complete_parameters = pd.DataFrame({
    "storm_draw": "storm_draw_" + parts[4],
    "source_id": parts[5].str.removeprefix("src"),
    "variant_label": parts[6].str.removeprefix("var"),
    "experiment_id": parts[7].str.removeprefix("exp"),
    "batch_year": parts[8].str.removeprefix("yr"),
    "basin": parts[9],
    "relative_risk": parts[10] + "_" + parts[11] + "_" + parts[12],
    "sample_name": parts[13] + "_" + parts[14],
})


rr_cols = ["indirect_cvd_draw", "indirect_resp_draw"]

final_long = full_tasks_df.melt(
    id_vars=[
        "source_id",
        "variant_label",
        "experiment_id",
        "batch_year",
        "basin",
        "storm_draw",
        "max_run_time",
        "memory_req",
        "num_cores",
    ],
    value_vars=rr_cols,
    var_name="relative_risk",
    value_name="sample_name",
)
# remaining_long = final_long.copy()

remaining_long = final_long.merge(
    complete_parameters,
    on=[
        "source_id",
        "variant_label",
        "experiment_id",
        "batch_year",
        "basin",
        "storm_draw",
        "relative_risk",
    ],
    how="left",
    indicator=True,
)

remaining_long = remaining_long[remaining_long["_merge"] == "left_only"].copy()

# # multiply max_run_time by 3
remaining_long["max_run_time"] = remaining_long["max_run_time"] * 3

# # multiply memory_req by 3
remaining_long["memory_req"] = remaining_long["memory_req"].str.replace("G", "").astype(int) * 3
remaining_long["memory_req"] = remaining_long["memory_req"].astype(str) + "G"

###############################################################
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
        "memory": "10G",
        "cores": 1,
        "runtime": "60m",
        "constraints": "archive",
        "queue": "all.q",
        "project": project,  # Ensure the project is set correctly
    }
)


# Get unique combinations of runtime, cores, and memory
unique_configs = remaining_long[['max_run_time', 'num_cores', 'memory_req']].drop_duplicates()

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
        default_resource_scales={
            "memory": 2,  # scale memory by 100% on retry
            "runtime": lambda x: int(x*1.25),  # scale runtime by 25%
        },
        max_attempts=5,
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

# delete sample_name_y
remaining_long = remaining_long.drop(columns=["sample_name_y", "_merge"])
# rename sample_name_x to sample_name
remaining_long = remaining_long.rename(columns={"sample_name_x": "sample_name"})


# Create tasks using the appropriate template
tasks = []
for row in remaining_long.itertuples():
    config_key = f"{row.max_run_time}_{row.num_cores}_{row.memory_req}"
    template = task_templates[config_key]

    task = template.create_task(
        name=(
            f"CLIMADA_stage3_"
            f"sd{row.storm_draw}_"
            f"src{row.source_id}_"
            f"var{row.variant_label}_"
            f"exp{row.experiment_id}_"
            f"yr{row.batch_year}_"
            f"{row.basin}_"
            f"{row.relative_risk}_"
            f"s{row.sample_name}_"
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
        relative_risk=row.relative_risk,
        sample_name=row.sample_name,
        num_cores=row.num_cores,
    )

    tasks.append(task)

print(f"Number of tasks: {len(tasks)}")
print(f"Number of task templates created: {len(task_templates)}")


###################################################################

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
