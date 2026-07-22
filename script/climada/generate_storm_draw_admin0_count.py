"""
Generate storm_draw_admin0_count_v2.parquet

For each (storm_draw, source_id, variant_label, experiment_id, batch_year, basin) task:
  - n_storms_in_batch: count of storm_XXXX dirs in the base stage1 zarr
  - estimated_storms_per_year: n_storms_in_batch / num_years_in_batch
  - num_admin0_first_year: unique admin0 regions in the first-year PAF raster
  - num_years_in_batch / estimated_admin0_total: derived

Output: /mnt/team/rapidresponse/pub/tropical-storms/climada/output/storm_draw_admin0_count_v2.parquet
"""

from pathlib import Path
from typing import Any

import gc

import geopandas as gpd  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import rasterra as rt  # type: ignore
from rasterio.features import shapes  # type: ignore
from rra_tools.parallel import run_parallel  # type: ignore
from shapely.geometry import (  # type: ignore
    GeometryCollection,
    MultiPolygon,
    Polygon,
    box,
    shape,
)
from shapely.ops import unary_union  # type: ignore

# ── Paths ────────────────────────────────────────────────────────────────────
TASK_ASSIGNMENTS_PATH = Path(
    "/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv"
)
STORM_DRAW_TABLE_PATH = Path(
    "/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv"
)
GDF_PATH = Path(
    "/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0.parquet"
)
PAF_ROOT = Path(
    "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage2_v2/"
)
STAGE1_ROOT = Path(
    "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1_v2"
)
OUTPUT_PATH = Path(
    "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/storm_draw_admin0_count.parquet"
)

NUM_CORES = 64


# ── Helper functions ─────────────────────────────────────────────────────────

def _polygonize(geom):
    if geom is None:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        return unary_union(polys) if polys else None
    return None


def subset_affected_area(rr_raster: rt.RasterArray, threshold: float = 0.0) -> rt.RasterArray:
    data = np.asarray(rr_raster.data)
    mask = np.isfinite(data) & (data > threshold)
    if not np.any(mask):
        return rr_raster

    rows, cols = np.where(mask)
    a, b, c, d, e, f = rr_raster.transform[:6]

    xmin = c + cols.min() * a
    xmax = c + (cols.max() + 1) * a
    ymax = f + rows.min() * e
    ymin = f + (rows.max() + 1) * e

    geom = box(xmin, ymin, xmax, ymax)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs=rr_raster.crs)
    return rr_raster.clip(gdf)


def load_raw_paf_raster(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    sample_name: str,
    relative_risk: str,
    year: int,
    paf_root: Path = PAF_ROOT,
) -> rt.RasterArray:
    start_year, end_year = batch_year.split("-")
    root_dir = (
        paf_root
        / storm_draw / source_id / variant_label / experiment_id
        / batch_year / str(year) / basin / "raw_paf"
    )
    filename = (
        f"draw_mean_raw_paf_{storm_draw}_{relative_risk}_{sample_name}_{basin}_"
        f"{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}.tif"
    )
    paf_path = root_dir / filename
    if not paf_path.exists():
        raise FileNotFoundError(f"PAF raster not found: {paf_path}")
    return rt.load_raster(paf_path)


def clean_paf_raster(raster: rt.RasterArray) -> rt.RasterArray:
    raster._ndarray = raster._ndarray.astype(np.float32, copy=False)
    raster._ndarray[raster._ndarray == 0] = np.nan
    return subset_affected_area(raster)


def intersect_shapefile_with_raster(
    shapefile_gdf: gpd.GeoDataFrame,
    rr_raster: rt.RasterArray,
    buffer_degrees: float = 0.0,
) -> gpd.GeoDataFrame:
    height, width = rr_raster._ndarray.shape
    transform = rr_raster.transform
    xmin, ymin = transform * (0, height)
    xmax, ymax = transform * (width, 0)
    raster_bbox = box(xmin, ymin, xmax, ymax)

    clipped = shapefile_gdf.clip(raster_bbox).copy()
    clipped["geometry"] = clipped.geometry.apply(_polygonize)
    clipped = clipped.dropna(subset=["geometry"]).reset_index(drop=True)

    rr_data = rr_raster._ndarray
    mask = rr_data > 0
    if not mask.any():
        return clipped.iloc[0:0].copy()

    geometries = [
        shape(geom).buffer(0)
        for geom, value in shapes(mask.astype(np.uint8), transform=rr_raster.transform)
        if value == 1
    ]
    rr_geom = unary_union(geometries)
    if isinstance(rr_geom, GeometryCollection):
        polys = [g for g in rr_geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        rr_geom = unary_union(polys)

    if buffer_degrees > 0:
        rr_geom = rr_geom.buffer(buffer_degrees)

    if clipped.crs.to_epsg() != 4326:
        clipped = clipped.to_crs("EPSG:4326")

    return clipped[clipped.intersects(rr_geom)].copy().reset_index(drop=True)


def count_storms_in_batch(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    stage1_root: Path = STAGE1_ROOT,
) -> int:
    start_year, end_year = batch_year.split("-")
    zarr_name = (
        f"intensity_{basin}_{source_id}_{experiment_id}_{variant_label}_"
        f"{start_year}01_{end_year}12.zarr"
    )
    zarr_path = (
        stage1_root / source_id / variant_label / experiment_id
        / batch_year / basin / "intensity" / zarr_name
    )
    if not zarr_path.exists():
        raise FileNotFoundError(f"Stage1 zarr not found: {zarr_path}")
    return sum(1 for f in zarr_path.iterdir() if f.is_dir() and f.name.startswith("storm_"))


# ── Build task dataframe ─────────────────────────────────────────────────────

def build_full_tasks_df() -> pd.DataFrame:
    meta_df = pd.read_csv(TASK_ASSIGNMENTS_PATH)
    meta_df = meta_df.drop(columns=["task_id", "draw"]).drop_duplicates()
    meta_df["basin"] = meta_df["basin"].fillna("NA")
    meta_df = meta_df.rename(columns={
        "model": "source_id",
        "variant": "variant_label",
        "scenario": "experiment_id",
        "time_period": "batch_year",
    })
    meta_df = meta_df[meta_df["batch_year"] != "1965-1969"].reset_index(drop=True)

    storm_draw_full = pd.read_csv(STORM_DRAW_TABLE_PATH)

    combo_cols = ["source_id", "variant_label"]
    sd = storm_draw_full[["storm_draw", "source_id", "variant_label"]].copy()
    sd["draw_rank"] = (
        sd.sort_values("storm_draw")
        .groupby(combo_cols)["storm_draw"]
        .rank(method="dense")
        .astype(int)
    )
    sd = sd.sort_values(["draw_rank", "source_id", "variant_label", "storm_draw"]).reset_index(drop=True)
    sd = sd.iloc[:8].drop(columns="draw_rank")

    tasks = sd.merge(meta_df, on=["source_id", "variant_label"], how="left")
    tasks = tasks.merge(storm_draw_full, on=["storm_draw", "source_id", "variant_label"], how="left")
    tasks["storm_draw"] = tasks["storm_draw"].apply(lambda x: f"storm_draw_{x:04d}")

    return tasks.reset_index(drop=True)


# ── Per-row processor ─────────────────────────────────────────────────────────

shapefile: gpd.GeoDataFrame = gpd.read_parquet(GDF_PATH)


def process_row(row: pd.Series) -> list[dict[str, Any]]:
    params = row.to_dict()
    start_year, end_year = map(int, params["batch_year"].split("-"))
    num_years = end_year - start_year + 1
    sample_name = params["indirect_cvd_draw"]

    raw_paf_raster = load_raw_paf_raster(
        storm_draw=params["storm_draw"],
        source_id=params["source_id"],
        variant_label=params["variant_label"],
        experiment_id=params["experiment_id"],
        batch_year=params["batch_year"],
        basin=params["basin"],
        sample_name=sample_name,
        relative_risk="indirect_cvd_draw",
        year=start_year,
    )
    raw_paf_raster = clean_paf_raster(raw_paf_raster)

    intersected = intersect_shapefile_with_raster(shapefile, raw_paf_raster)
    num_admin0_first_year = intersected["ADM0_CODE"].nunique()

    del raw_paf_raster, intersected
    gc.collect()

    n_storms = count_storms_in_batch(
        source_id=params["source_id"],
        variant_label=params["variant_label"],
        experiment_id=params["experiment_id"],
        batch_year=params["batch_year"],
        basin=params["basin"],
    )

    return [{
        **params,
        "n_storms_in_batch": n_storms,
        "estimated_storms_per_year": n_storms / num_years,
        "year": str(start_year),
        "num_admin0_first_year": num_admin0_first_year,
        "num_years_in_batch": num_years,
        "estimated_admin0_total": num_admin0_first_year * num_years,
    }]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building task list...")
    full_tasks_df = build_full_tasks_df()
    print(f"Total tasks: {len(full_tasks_df)}")

    rows = [row for _, row in full_tasks_df.iterrows()]

    print(f"Processing {len(rows)} rows with {NUM_CORES} cores...")
    results_nested = run_parallel(
        runner=process_row,
        arg_list=rows,
        num_cores=NUM_CORES,
        progress_bar=True,
    )

    results = [item for sublist in results_nested for item in sublist]
    df = pd.DataFrame(results)

    print(f"Saving {len(df)} rows to {OUTPUT_PATH}")
    df.to_parquet(OUTPUT_PATH, index=False)
    OUTPUT_PATH.chmod(0o775)
    print("Done.")
