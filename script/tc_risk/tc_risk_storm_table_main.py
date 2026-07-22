from pathlib import Path
import pandas as pd
import xarray as xr
import os
import argparse

parser = argparse.ArgumentParser(description="TC Risk Storm Table Generator")

# Define arguments
parser.add_argument("--source_id", type=str, required=True, help="Source Id")
parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")

# Parse arguments
args = parser.parse_args()
source_id = args.source_id
variant_label = args.variant_label
experiment_id = args.experiment_id

ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/")
DRAWS = list(range(0, 100))

def generate_metadata():
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

    return meta_df

def save_storm_table(
    storm_table_df: pd.DataFrame,
    source_id: str,
    variant_label: str,
    experiment_id: str,
) -> None:

    output_dir = ROOT_PATH / "storm_tables" / source_id / variant_label / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"storm_table_{source_id}_{variant_label}_{experiment_id}.parquet"
    storm_table_df.to_parquet(output_path, index=False)
    os.chmod(output_path, 0o775)

    

def read_custom_tracks_nc(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
) -> Path:

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

    return nc_file

def generate_storm_table(
    source_id: str,
    variant_label: str,
    experiment_id: str,
) -> pd.DataFrame:

    meta_df = generate_metadata()

    meta_df_filtered = meta_df[
        (meta_df["source_id"] == source_id) &
        (meta_df["variant_label"] == variant_label) &
        (meta_df["experiment_id"] == experiment_id)
    ].reset_index(drop=True)

    storm_tables = []
    for _, row in meta_df_filtered.iterrows():
        for draw in DRAWS:
            ds_path = read_custom_tracks_nc(
                source_id=row["source_id"],
                variant_label=row["variant_label"],
                experiment_id=row["experiment_id"],
                batch_year=row["batch_year"],
                basin=row["basin"],
                draw=draw,
            )

            with xr.open_dataset(
                ds_path,
                engine="netcdf4",
                chunks={"n_trk": 10},  # adjust chunk size as needed
                decode_times=False,  # keep time as numeric for now
            ) as ds:
                
                n_trk = ds.sizes.get("n_trk", 0)
                if n_trk == 0:
                    print(f"No storms found in dataset: {ds_path}")
                    continue
            
                # reduce wind over time dimension to get max wind per storm
                max_wind = ds["max_sustained_wind"].max(dim="time")

                assert ds["tc_years"].sizes["n_trk"] == n_trk, f"Expected n_trk size {n_trk} but got {ds['tc_years'].sizes['n_trk']}"
                assert ds["tc_month"].sizes["n_trk"] == n_trk, f"Expected n_trk size {n_trk} but got {ds['tc_month'].sizes['n_trk']}"
                assert max_wind.sizes["n_trk"] == n_trk, f"Expected n_trk size {n_trk} but got {max_wind.sizes['n_trk']}"

                df = pd.DataFrame({
                    "source_id": row["source_id"],
                    "variant_label": row["variant_label"],
                    "experiment_id": row["experiment_id"],
                    "batch_year": row["batch_year"],
                    "basin": row["basin"],
                    "year": ds["tc_years"].values,
                    "month": ds["tc_month"].values,
                    "tc_risk_draw": draw,
                    "storm_id": range(n_trk),
                    "max_sustained_wind": max_wind.values,
                })

                storm_tables.append(df)

    storm_table_df = pd.concat(storm_tables, ignore_index=True)

    save_storm_table(
        storm_table_df=storm_table_df,
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
    )
    
generate_storm_table(
    source_id=source_id,
    variant_label=variant_label,
    experiment_id=experiment_id,
)