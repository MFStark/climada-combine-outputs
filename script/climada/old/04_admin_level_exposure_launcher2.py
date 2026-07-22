import getpass
from importlib.util import source_from_cache
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
import os
import sys

# DRAW_BATCHES = [
# "0-4",
# "5-9",
# "10-14",
# "15-19",
# "20-24",
# "25-29",
# "30-34",
# "35-39",
# "40-44",
# "45-49",
# "50-54",
# "55-59",
# "60-64",
# "65-69",
# "70-74",
# "75-79",
# "80-84",
# "85-89",
# "90-94",
# "95-99",
# ]

# Draw batches of 2 instead of 5 for testing
# DRAW_BATCHES = [
# "0-1",
# "2-3",
# "4-5",
# "6-7",
# "8-9",  
# "10-11",
# "12-13",
# "14-15",
# "16-17",
# "18-19",
# "20-21",
# "22-23",
# "24-25",
# "26-27",
# "28-29",
# "30-31",
# "32-33",
# "34-35",
# "36-37",
# "38-39",
# "40-41",
# "42-43",
# "44-45",
# "46-47",
# "48-49",
# "50-51",
# "52-53",
# "54-55",
# "56-57",
# "58-59",
# "60-61",
# "62-63",
# "64-65",
# "66-67",
# "68-69",
# "70-71",
# "72-73",
# "74-75",
# "76-77",
# "78-79",
# "80-81",
# "82-83",
# "84-85",
# "86-87",
# "88-89",
# "90-91",
# "92-93",
# "94-95",
# "96-97",
# "98-99",
# ]

# Draw batches of 1 for testing
DRAW_BATCHES = [f"{i}-{i}" for i in range(100)]

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

# drop 1965-1969 batch
meta_df = meta_df[meta_df["batch_year"] != "1965-1969"].reset_index(drop=True)

# read in resource requirements df
resource_df = pd.read_parquet("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4_resource_requirements.parquet")
resource_df = resource_df.drop(columns=["year", "num_admin0_first_year", "num_years_in_batch", "estimated_admin0_total", "draw_batch"], errors="ignore")


meta_df = meta_df.merge(resource_df, on=["source_id", "variant_label", "experiment_id", "batch_year", "basin"], how="left")

# fill any missing required run time with 5.0 and missing memory with 16.0
meta_df["req_runtime_min"] = meta_df["req_runtime_min"].fillna(5.0)
meta_df["req_mem_gb_rounded"] = meta_df["req_mem_gb_rounded"].fillna(16.0)

# assign column num_cores = 5
meta_df["num_cores"] = 1

# multiply required memory by 5 for parallelization
meta_df["memory_req"] = meta_df["req_mem_gb_rounded"].apply(lambda x: f"{int(x * 1.25)}G")

full_tasks = (meta_df
    .assign(key=1)
    .merge(pd.DataFrame({"draw_batch": DRAW_BATCHES, "key": 1}), on="key")
    .drop(columns=["key"])
)


# subset to one model for testing
# source_id = "EC-Earth3"
# variant_label = "r1i1p1f1"
source_id = "MRI-ESM2-0"
variant_label = "r2i1p1f1"

full_tasks = full_tasks[
    (full_tasks["source_id"] == source_id) &
    (full_tasks["variant_label"] == variant_label)
].reset_index(drop=True)

# chnage min run time to 60m for testing
full_tasks["req_runtime_min"] = 30.0

# change memory to 60G for testing
full_tasks["memory_req"] = "50G"

##########################################################
workflow_id1 = 562139
workflow_id2 = 562141
workflow_id3 = 562224


workflow_ids = [workflow_id1, workflow_id2, workflow_id3]

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
    "source_id": parts[2].str.removeprefix("src"),
    "variant_label": parts[3].str.removeprefix("var"),
    "experiment_id": parts[4].str.removeprefix("exp"),
    "batch_year": parts[5].str.removeprefix("yr"),
    "basin": parts[6],
    "draw_batch": parts[10].str.removeprefix("db"),
})


complete_parameters = complete_parameters.merge(full_tasks, on=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw_batch"], how="inner")


# drop complete parameters from full tasks to get remaining tasks

remaining_meta = full_tasks.merge(complete_parameters[["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw_batch"]], on=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw_batch"], how="left", indicator=True)
remaining_meta = remaining_meta[remaining_meta["_merge"] == "left_only"].drop(columns=["_merge"])



full_tasks = remaining_meta.copy()


#######################################################################
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
unique_configs = full_tasks[['req_runtime_min', 'num_cores', 'memory_req']].drop_duplicates()

# Create task templates for each unique configuration
task_templates = {}
for _, config in unique_configs.iterrows():
    config_key = f"{config['req_runtime_min']}_{config['num_cores']}_{config['memory_req']}"
    
    task_templates[config_key] = tool.get_task_template(
        template_name=f"CLIMADA_stage4_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": config['num_cores'],
            "memory": config['memory_req'],
            "runtime": f"{int(config['req_runtime_min'])}m",
            "project": project,
        },
        default_resource_scales={
            "memory": lambda x: int(x*1.5),  # scale memory by 50%
            "runtime": lambda x: int(x*2),  # scale runtime by 100%
        },
        max_attempts=5,
        command_template=(
            "python /ihme/homes/mfiking/github_repos/climada_python/script/climada/04_admin_level_exposure_main_04_02_26.py "
            "--source_id {source_id} "
            "--variant_label {variant_label} "
            "--experiment_id {experiment_id} "
            "--batch_year {batch_year} "
            "--basin {basin} "
            "--draw_batch {draw_batch} "
            "--num_cores {num_cores}"
        ),
        node_args=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw_batch", "num_cores"],
    )


# Create tasks using the appropriate template
tasks = []
for row in full_tasks.itertuples():
    config_key = f"{row.req_runtime_min}_{row.num_cores}_{row.memory_req}"
    template = task_templates[config_key]

    task = template.create_task(
        name=(
            f"CLIMADA_stage4_"
            f"src{row.source_id}_"
            f"var{row.variant_label}_"
            f"exp{row.experiment_id}_"
            f"yr{row.batch_year}_"
            f"{row.basin}_"
            f"rt{row.req_runtime_min}m_"
            f"mem{row.memory_req}_"
            f"cores{row.num_cores}_"
            f"db{row.draw_batch}"
        ),
        source_id=row.source_id,
        variant_label=row.variant_label,
        experiment_id=row.experiment_id,
        batch_year=row.batch_year,
        basin=row.basin,
        draw_batch=row.draw_batch,
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
