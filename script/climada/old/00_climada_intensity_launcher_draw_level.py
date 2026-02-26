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



ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")


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
    task_df_copy = task_df.copy()
    task_df_copy = task_df_copy[["source_id", "variant_label", "experiment_id", "batch_year", "basin"]].drop_duplicates().reset_index(drop=True)
    task_df_copy["num_tracks"] = run_parallel(
        runner=single_storm_count,
        arg_list=[row for _, row in task_df_copy.iterrows()],
        num_cores=24,
        progress_bar=True,
    )

    # remerge num_tracks back to original task_df
    task_df_with_counts = task_df.merge(
        task_df_copy[["source_id", "variant_label", "experiment_id", "batch_year", "basin", "num_tracks"]],
        on=["source_id", "variant_label", "experiment_id", "batch_year", "basin"],
        how="left",
    )

    return task_df_with_counts


def assign_run_time_single_draw(row: pd.Series) -> pd.Series:
    """
    Assigns runtime and resources for single-draw reruns.
    - Runtime scales roughly 1 minute per storm track.
    - Adds a fixed buffer for overhead.
    - Fixed resources: 1 core, 15 GB memory.
    """
    num_tracks = row["num_tracks"]

    # Fixed resources
    base_cores = 1
    base_memory_gb = 17.5
    runtime_per_track = 1       # 1 minute per track
    runtime_buffer = 30         # fixed overhead in minutes
    min_runtime = 35            # optional minimum runtime

    # Calculate runtime
    estimated_runtime = num_tracks * runtime_per_track
    estimated_runtime = max(estimated_runtime, min_runtime)

    # Add buffer
    final_runtime = estimated_runtime + runtime_buffer

    # Assign
    row["max_run_time"] = final_runtime
    row["num_cores"] = base_cores
    row["memory_req"] = f"{base_memory_gb}G"

    return row


# Read in paths
meta_df = pd.read_csv("/mnt/share/homes/mfiking/downloads/jobmon_workflows/climada_stage0/error_partial_zarrs.csv")
meta_df = meta_df.drop(columns=["path", "metric"]).drop_duplicates()

# get storm counts
meta_df_storm_counts = run_storm_count_parallel(meta_df)

# Assign run times based on storm counts
meta_df_storm_counts = meta_df_storm_counts.apply(assign_run_time_single_draw, axis=1)

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
unique_configs = meta_df_storm_counts[['max_run_time', 'num_cores', 'memory_req']].drop_duplicates()

# Create task templates for each unique configuration
task_templates = {}
for _, config in unique_configs.iterrows():
    config_key = f"{config['max_run_time']}_{config['num_cores']}_{config['memory_req']}"
    
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
            "--draw {draw}"
        ),
        node_args=["source_id", "variant_label", "experiment_id", "batch_year", "basin", "draw"],
        task_args=[],
        op_args=[],
    )

# Create tasks using the appropriate template
tasks = []
for row in meta_df_storm_counts.itertuples():
    config_key = f"{row.max_run_time}_{row.num_cores}_{row.memory_req}"
    template = task_templates[config_key]
    
    task = template.create_task(
        name=f"CLIMADA_stage0_{row.source_id}_{row.variant_label}_{row.experiment_id}_{row.batch_year}_{row.basin}_d{row.draw_batch}",
        source_id=row.source_id,
        variant_label=row.variant_label,
        experiment_id=row.experiment_id,
        batch_year=row.batch_year,
        basin=row.basin,
        draw=row.draw,
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
