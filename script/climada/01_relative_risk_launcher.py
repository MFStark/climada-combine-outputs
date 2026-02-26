import getpass
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
import os
import sys
from rra_tools.parallel import run_parallel # type: ignore
import xarray as xr # type: ignore

RELATIVE_RISKS = ["indirect_resp_draw", "indirect_cvd_draw"]

DRAW_BATCHES = [
    "0-49",
    "50-99",
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
    return task_df


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
# replace nan with NA
meta_df = meta_df.fillna("NA")

# Normalize column names
meta_df = meta_df.rename(columns={
    "model": "source_id",
    "variant": "variant_label",
    "scenario": "experiment_id",
    "time_period": "batch_year",
})


# subset to test
source_id = "MRI-ESM2-0"
variant_label = "r1i1p1f1"
experiment_id = "historical"
batch_year = "2011-2014"
basin = "WP"
meta_df = meta_df[
    (meta_df["source_id"] == source_id) &
    (meta_df["variant_label"] == variant_label) &
    (meta_df["experiment_id"] == experiment_id) &
    (meta_df["batch_year"] == batch_year) &
    (meta_df["basin"] == basin)
].reset_index(drop=True)



# get counts of storms per source_id, variant_label, experiment_id, batch_year, basin
meta_df_storm_counts = run_storm_count_parallel(meta_df)

# read in storm draws
storm_draw_df = pd.read_csv("/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv")

complete_df = meta_df_storm_counts.merge(
    storm_draw_df,
    on=["source_id", "variant_label",],
    how="inner",
)

# replace storm_draw as storm_draw_XXXX
complete_df["storm_draw"] = complete_df["storm_draw"].apply(lambda x: f"storm_draw_{x:04d}")

# Assign run times based on storm counts
meta_df_storm_counts = complete_df.apply(assign_run_time, axis=1)

# take first storm_draw for testing
storm_draws = meta_df_storm_counts["storm_draw"].unique()
first_draw = storm_draws[0]

# Create full tasks by cross-joining with draw batches
full_tasks_df = (
    meta_df_storm_counts
    .assign(key=1)
    .merge(pd.DataFrame({"draw_batch": DRAW_BATCHES, "key": 1}), on="key")
    .drop(columns=["key"])
)

full_tasks_df = full_tasks_df[full_tasks_df["storm_draw"] == first_draw].reset_index(drop=True)

user = getpass.getuser()

# Project
project = "proj_rapidresponse"  # Adjust this to your project name if needed

# create jobmon jobs
user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="CLIMADA_stage1")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"CLIMADA_stage1_{wf_uuid}",
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
        template_name=f"CLIMADA_stage1_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": config['num_cores'],
            "memory": config['memory_req'],
            "runtime": f"{int(config['max_run_time'])}m",
            "project": project,
        },
        command_template=(
            "python /ihme/homes/mfiking/github_repos/climada_python/script/climada/01_relative_risk_main.py "
            "--storm_draw {storm_draw} "
            "--source_id {source_id} "
            "--variant_label {variant_label} "
            "--experiment_id {experiment_id} "
            "--batch_year {batch_year} "
            "--basin {basin} "
            "--relative_risk {relative_risk} "
            "--sample_name {sample_name} "
            "--draw_batch {draw_batch} "
            "--num_cores {num_cores}"
        ),
        node_args=["storm_draw", "source_id", "variant_label", "experiment_id", "batch_year", "basin", "relative_risk", "sample_name", "draw_batch", "num_cores"],
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
            name=f"CLIMADA_stage1_{row.storm_draw}_{row.source_id}_{row.variant_label}_{row.experiment_id}_{row.batch_year}_{row.basin}_{relative_risk}_{sample_name}_d{row.draw_batch}_c{row.num_cores}",
            storm_draw=row.storm_draw,
            source_id=row.source_id,
            variant_label=row.variant_label,
            experiment_id=row.experiment_id,
            batch_year=row.batch_year,
            basin=row.basin,
            relative_risk=relative_risk,
            sample_name=sample_name,
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
