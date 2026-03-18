import getpass
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
    n_admin0 = row["num_admin0_first_year"]
    n_years = row["num_years_in_batch"]

    # --- Runtime estimation ---
    slope_per_admin0 = (65.8 - 4.7) / (43 - 8)
    base_runtime_for_5yrs = 4.7 - slope_per_admin0 * 8

    runtime_min = (base_runtime_for_5yrs + slope_per_admin0 * n_admin0) * (n_years / 5)

    # enforce minimum runtime
    runtime_min = max(runtime_min, 4)

    # round to nearest 5 minutes
    runtime_rounded = int(round(runtime_min / 5) * 5)
    runtime_rounded = max(runtime_rounded, 5)

    # --- Memory estimation ---
    if n_admin0 <= 8:
        memory_gb = 20
    elif n_admin0 <= 21:
        memory_gb = 20 + (26 - 20) * (n_admin0 - 8) / (21 - 8)
    elif n_admin0 <= 43:
        memory_gb = 26 + (61 - 26) * (n_admin0 - 21) / (43 - 21)
    else:
        memory_gb = 61 + (n_admin0 - 43) * (61 - 26) / (43 - 21)

    # round to nearest 4 GB
    memory_rounded = int(round(memory_gb / 4) * 4)
    memory_rounded = max(memory_rounded, 4)

    row["memory_req"] = f"{memory_rounded}G"

    row["num_cores"] = 1
    row["max_run_time"] = runtime_rounded
    row["memory_req"] = f"{memory_rounded}G" #modify meory usage by 5 times 

    return row

########################################
#    Get completed workflow tasks      #
########################################
# workflow_id1 = 552899
# workflow_id2 = 552932
# workflow_id3 = 553088

# df1 = workflow_tasks(
#     workflow_id=workflow_id1,
#     limit=-1   # return all tasks
# )
# df2 = workflow_tasks(
#     workflow_id=workflow_id2,
#     limit=-1   # return all tasks
# )
# df3 = workflow_tasks(
#     workflow_id=workflow_id3,
#     limit=-1   # return all tasks
# )

# completed_df1 = df1[df1["STATUS"] == "DONE"]
# completed_df2 = df2[df2["STATUS"] == "DONE"]
# completed_df3 = df3[df3["STATUS"] == "DONE"]

# df = pd.concat([completed_df1, completed_df2, completed_df3])

# # Create completed parameters df
# parts = df["TASK_NAME"].str.split("_", expand=True)

# complete_parameters = pd.DataFrame({
#     "storm_draw": "storm_" + parts[3] + "_" + parts[4],   # storm_draw_XXXX
#     "source_id": parts[5].str.replace("src","",regex=False),
#     "variant_label": parts[6].str.replace("var","",regex=False),
#     "experiment_id": parts[7].str.replace("exp","",regex=False),
#     "batch_year": parts[8].str.replace("yr","",regex=False),
#     "basin": parts[9],
#     "relative_risk": parts[10] + "_" + parts[11] + "_" + parts[12],
#     "sample_name": parts[13] + "_" + parts[14]
# })


################################
# Get metadata for all tasks   #
################################
meta_df = pd.read_parquet("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/storm_draw_admin0_count.parquet")

# Convert to long format
id_cols = [
    "storm_draw",
    "source_id",
    "variant_label",
    "experiment_id",
    "batch_year",
    "basin",
    "year",
    "num_admin0_first_year",
    "num_years_in_batch",
    "estimated_admin0_total",
]

risk_cols = [
    "indirect_cvd_draw",
    "indirect_resp_draw",
]

meta_long = (
    meta_df
    .melt(
        id_vars=id_cols,
        value_vars=risk_cols,
        var_name="relative_risk",
        value_name="sample_name",
    )
    .reset_index(drop=True)
)

meta_long = meta_long[
    [
        "storm_draw",
        "source_id",
        "variant_label",
        "experiment_id",
        "batch_year",
        "basin",
        "relative_risk",
        "sample_name",
        "year",
        "num_admin0_first_year",
        "num_years_in_batch",
        "estimated_admin0_total",
    ]
]

# Specific failed tasks
root = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage3_log")

df_list = []

for file in root.iterdir():
    df = pd.read_parquet(file)
    df_list.append(df)
df_full = pd.concat(df_list)
df_full = df_full.drop(columns=["year", "error_type"])

df_full = df_full.merge(meta_long, on=["storm_draw", "source_id", "variant_label", "experiment_id", "batch_year", "basin", "relative_risk", "sample_name"], how="inner")



######################################################
# Filter original metadata with completed parameters #
######################################################
# key_cols = [
#     "storm_draw",
#     "source_id",
#     "variant_label",
#     "experiment_id",
#     "batch_year",
#     "basin",
#     "relative_risk",
#     "sample_name",
# ]

# remaining_meta = (
#     meta_long
#     .merge(
#         complete_parameters[key_cols].drop_duplicates(),
#         on=key_cols,
#         how="left",
#         indicator=True
#     )
#     .query('_merge == "left_only"')
#     .drop(columns="_merge")
#     .reset_index(drop=True)
# )

#########################################
#  Assign resources to remaining tasks  #
#########################################

full_tasks_df = df_full.apply(assign_resources_single_core, axis=1)


# Assign run times based on storm counts
# full_tasks_df = remaining_meta.apply(assign_resources_single_core, axis=1)

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

# TEST
# full_tasks_df = full_tasks_df[full_tasks_df["num_years_in_batch"] == 3][:1]
############################################

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
    name=f"CLIMADA_stage3{wf_uuid}",
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
        template_name=f"CLIMADA_stage3_{config_key}",
        default_cluster_name="slurm",
        default_compute_resources={
            "queue": "all.q",
            "cores": config['num_cores'],
            "memory": config['memory_req'],
            "runtime": f"{int(config['max_run_time'])}m",
            "project": project,
        },
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


# Create tasks using the appropriate template
tasks = []
for row in full_tasks_df.itertuples():
    config_key = f"{row.max_run_time}_{row.num_cores}_{row.memory_req}"
    template = task_templates[config_key]
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
            f"tracks_per_year{row.num_admin0_first_year}_"
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
