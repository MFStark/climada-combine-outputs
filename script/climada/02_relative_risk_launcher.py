import getpass
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path

from zarr import full
from rra_tools.parallel import run_parallel # type: ignore
import xarray as xr # type: ignore
import numpy as np
import math 

RELATIVE_RISKS = ["indirect_resp_draw", "indirect_cvd_draw"]

ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")

# Read in paths
meta_df = pd.read_csv("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv")
meta_df = meta_df.drop(columns=["task_id", "draw"]).drop_duplicates()
# replace nan with NA
meta_df = meta_df.fillna("NA")

# Normalize column names
meta_df = meta_df.rename(columns={
    "model": "source_id",
    "variant": "variant_label",
    "scenario": "experiment_id",
    "time_period": "batch_year",
})

meta_df = meta_df[meta_df["batch_year"] != "1965-1969"]

# read in storm draws
storm_draw_df = pd.read_csv("/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv")

complete_df = meta_df.merge(
    storm_draw_df, 
    on=["source_id", "variant_label"],
    how="inner",
)

# replace storm_draw as storm_draw_XXXX
complete_df["storm_draw"] = complete_df["storm_draw"].apply(lambda x: f"storm_draw_{x:04d}")

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

priority_map = {v: i for i, v in enumerate(priority)}

df = complete_df.copy()

df["draw_order"] = df["storm_draw"].map(priority_map)

# fill non-priority draws with large value → go last
df["draw_order"] = df["draw_order"].fillna(len(priority))

df = df.sort_values(["draw_order", "storm_draw", "source_id", "variant_label", "experiment_id", "batch_year", "basin"]).drop(columns="draw_order")

# # subset to priority storm draws for testing
# df = df[df["storm_draw"].isin(priority)]

# subset to not the priority storm draws for complete run
df = df[~df["storm_draw"].isin(priority)]

#########################################################################################
# assign runtime
resource_df = pd.read_parquet("/mnt/share/homes/mfiking/downloads/climada_rs/stage2_resource_usage.parquet")
resource_df = resource_df.drop(columns=["task_id", "runtime", "memory", "memory_gb", "memory_rounded"])
resource_df = resource_df.rename(columns={
    "runtime_rounded": "max_run_time",
})

# assign 4GB
resource_df["memory_req"] = "4G"
# assign 10 cores
resource_df["num_cores"] = 10
resource_df = resource_df.drop_duplicates(subset=["source_id", "variant_label", "experiment_id", "batch_year", "basin"])

# merge with main df
final_df = df.merge(
    resource_df,
    on=["source_id", "variant_label", "experiment_id", "batch_year", "basin"],
    how="left"
)
# fill any na maxrun_time with 60 minutes, memory_req with 4G, num_cores with 10
final_df["max_run_time"] = final_df["max_run_time"].fillna(30)
final_df["memory_req"] = final_df["memory_req"].fillna("4G")
final_df["num_cores"] = final_df["num_cores"].fillna(10)


# round runtimes to 300, 600, 1200, 1800, 2700, 3600, 5700
def round_runtime(x):
    if x <= 300:
        return 300
    elif x <= 600:
        return 600
    elif x <= 1200:
        return 1200
    elif x <= 1800:
        return 1800
    elif x <= 2700:
        return 2700
    elif x <= 3600:
        return 3600
    else:
        return 5700
    
final_df["max_run_time"] = final_df["max_run_time"].apply(round_runtime)
# divide max runtimes by 60 to convert to minutes and round up
final_df["max_run_time"] = np.ceil(final_df["max_run_time"] / 60).astype(int)


#########################################################################################
workflow_id1 = 557853
workflow_id2 = 557918
workflow_id3 = 558362
workflow_id4 = 558593
workflow_id5 = 558645

workflow_ids = [workflow_id1, workflow_id2, workflow_id3, workflow_id4, workflow_id5]

total_df_list = []

for workflow_id in workflow_ids:
    df = workflow_tasks(
        workflow_id=workflow_id,
        limit=-1   # return all tasks
    )
    completed_df = df[df["STATUS"].isin(["COMPLETED", "D", "DONE", "Done"])]
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

final_long = final_df.melt(
    id_vars=[
        "source_id",
        "variant_label",
        "experiment_id",
        "batch_year",
        "basin",
        "storm_draw",
        "max_run_time",
        "runtime_min",
        "memory_req",
        "num_cores",
    ],
    value_vars=rr_cols,
    var_name="relative_risk",
    value_name="sample_name",
)

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
        "sample_name",
    ],
    how="left",
    indicator=True,
)

remaining_long = remaining_long[remaining_long["_merge"] == "left_only"].copy()

# multiply max_run_time by 3
remaining_long["max_run_time"] = remaining_long["max_run_time"] * 3


############################################################################################

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
        template_name=f"CLIMADA_stage2_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": config['num_cores'],
            "memory": config['memory_req'],
            "runtime": f"{int(config['max_run_time'])}m",
            "project": project,
        },
        command_template=(
            "python /ihme/homes/mfiking/github_repos/climada_python/script/climada/02_relative_risk_main.py "
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
for row in remaining_long.itertuples():
    config_key = f"{row.max_run_time}_{row.num_cores}_{row.memory_req}"
    template = task_templates[config_key]

    # for relative_risk in RELATIVE_RISKS:
    #     if relative_risk == "indirect_resp_draw":
    #         sample_name = row.indirect_resp_draw
    #     elif relative_risk == "indirect_cvd_draw":
    #         sample_name = row.indirect_cvd_draw
    #     else:
    #         raise ValueError(f"Unexpected relative risk type: {relative_risk}")

    task = template.create_task(
        name=(
            f"CLIMADA_stage2_"
            f"sd{row.storm_draw}_"
            f"src{row.source_id}_"
            f"var{row.variant_label}_"
            f"exp{row.experiment_id}_"
            f"yr{row.batch_year}_"
            f"{row.basin}_"
            f"{row.relative_risk}_"
            f"{row.sample_name}_"
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
