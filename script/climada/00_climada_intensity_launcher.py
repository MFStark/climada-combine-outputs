import getpass
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path
import xarray as xr  # type: ignore
from rra_tools.parallel import run_parallel  # type: ignore
import os

DRAW_BATCHES = [
# "0-4",
# "0-0",
"1-1",
# "2-2",
# "3-3",
# "4-4",
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
]

ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")
def parse_task_name(df: pd.DataFrame) -> pd.DataFrame:
    # Split the task_name into components
    components = df["task_name"].str.split("_", expand=True)
    
    # Assign components to new columns
    task_df = df.copy()
    task_df["source_id"] = components[2]
    task_df["variant_label"] = components[3]
    task_df["experiment_id"] = components[4]
    task_df["batch_year"] = components[5]
    task_df["basin"] = components[6]
    task_df["draw_batch"] = components[7].str[1:]
    return task_df[["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw_batch"]]

def read_custom_tracks_nc(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int = 0,
) -> xr.Dataset:

    start_year, end_year = batch_year.split("-")
    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    nc_file = (
        ROOT_PATH
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / f"tracks_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.nc"
    )

    if not nc_file.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_file}")

    return xr.open_dataset(nc_file)

def count_tracks(ds: xr.Dataset) -> int:
    return ds.sizes["n_trk"]

def single_storm_count(row: pd.Series) -> int:
    ds = read_custom_tracks_nc(
        source_id=row["source_id"],
        variant_label=row["variant_label"],
        experiment_id=row["experiment_id"],
        batch_year=row["batch_year"],
        basin=row["basin"],
    )
    num_tracks = count_tracks(ds)
    ds.close()
    return num_tracks

def run_storm_count_parallel(task_df: pd.DataFrame) -> pd.DataFrame:
    task_df = task_df.copy()
    task_df = task_df[["source_id", "variant_label", "experiment_id", "batch_year", "basin"]].drop_duplicates().reset_index(drop=True)
    task_df["num_tracks"] = run_parallel(
        runner=single_storm_count,
        arg_list=[row for _, row in task_df.iterrows()],
        num_cores=10,
        progress_bar=True,
    )
    return task_df[["source_id", "variant_label", "experiment_id", "batch_year", "basin", "num_tracks"]]

def assign_run_time(row: pd.Series) -> pd.Series:
    num_tracks = row["num_tracks"]

    # Fixed resources per task
    row["num_cores"] = 5
    row["memory_req"] = "35G"

    # Runtime bins based on empirical performance
    if num_tracks <= 15:
        base_runtime = 40
    elif num_tracks <= 30:
        base_runtime = 50
    elif num_tracks <= 70:
        base_runtime = 80
    elif num_tracks <= 120:
        base_runtime = 120
    elif num_tracks <= 180:
        base_runtime = 190
    else:
        base_runtime = 250

    # Safety buffer (protects against node variance)
    runtime_buffer = 10
    row["max_run_time"] = base_runtime + runtime_buffer

    return row


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

# test one model 
source_id = "CMCC-ESM2"
variant_label = "r1i1p1f1"
basin = "EP"

meta_df = meta_df[
    (meta_df["source_id"] == source_id) &
    (meta_df["variant_label"] == variant_label) 
]
historical_df = meta_df[meta_df["batch_year"] == "2011-2014"]
ssp126_df = meta_df[meta_df["batch_year"] == "2070-2074"]
ssp245_df = meta_df[meta_df["batch_year"] == "2029-2032"]
ssp585_df = meta_df[meta_df["batch_year"] == "2046-2048"]

sample_df = pd.concat([historical_df, ssp126_df, ssp245_df, ssp585_df])

# test failed CLIMADA_stage0_CMCC-ESM2_r1i1p1f1_ssp245_2029-2032_NI_d0-4_c5 
sample_df = sample_df[
    (sample_df["source_id"] == "CMCC-ESM2") & 
    (sample_df["variant_label"] == "r1i1p1f1") &
    (sample_df["experiment_id"] == "ssp245") &
    (sample_df["batch_year"] == "2029-2032") & 
    (sample_df["basin"] == "NI")
]

# get storm counts
meta_df_storm_counts = run_storm_count_parallel(sample_df)

# Assign run times based on storm counts
meta_df_storm_counts = meta_df_storm_counts.apply(assign_run_time, axis=1)

# # Create full tasks by cross-joining with draw batches
full_tasks_df = (
    meta_df_storm_counts[["source_id", "variant_label", "experiment_id", "batch_year", "basin", "num_tracks", "max_run_time", "num_cores", "memory_req"]]
    .assign(key=1)
    .merge(pd.DataFrame({"draw_batch": DRAW_BATCHES, "key": 1}), on="key")
    .drop(columns=["key"])
)



user = getpass.getuser()

# Project
project = "proj_rapidresponse"  # Adjust this to your project name if needed

# create jobmon jobs
user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="CLIMADA_stage0")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"CLIMADA_stage0_{wf_uuid}",
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

# Get unique combinations of runtime, cores, and memory
unique_configs = full_tasks_df[['max_run_time', 'num_cores', 'memory_req']].drop_duplicates()

# Create task templates for each unique configuration
task_templates = {}
for _, config in unique_configs.iterrows():
    config_key = f"rt{config['max_run_time']}_c{config['num_cores']}_m{config['memory_req']}"
    
    task_templates[config_key] = tool.get_task_template(
        template_name=f"CLIMADA_stage0_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": config['num_cores'],
            "memory": config['memory_req'],
            "runtime": f"{int(config['max_run_time'])}m",
            "project": project,
        },
        command_template=(
            "python /ihme/homes/mfiking/github_repos/climada_python/script/climada/00_climada_intensity_main.py "
            "--source_id {source_id} "
            "--variant_label {variant_label} "
            "--experiment_id {experiment_id} "
            "--batch_year {batch_year} "
            "--basin {basin} "
            "--draw_batch {draw_batch} "
            "--num_cores {num_cores} "
        ),
        node_args=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw_batch", "num_cores"],
        task_args=[],
        op_args=[],
    )

# Create tasks using the appropriate template
tasks = []
for row in full_tasks_df.itertuples():
    config_key = f"rt{row.max_run_time}_c{row.num_cores}_m{row.memory_req}"
    template = task_templates[config_key]
    
    task = template.create_task(
        name=f"CLIMADA_stage0_{row.source_id}_{row.variant_label}_{row.experiment_id}_{row.batch_year}_{row.basin}_d{row.draw_batch}_c{row.num_cores}",
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
