# CLIMADA Pipeline — Stage Reference

This document tracks each stage of the CLIMADA tropical cyclone processing
pipeline: purpose, inputs, outputs, control flow, and operational behavior.
Update one section per stage as the pipeline evolves.

---

## Data versions

The pipeline pins three external datasets. Each is referenced by every
stage 3 + 4A + 4B main script through a module-level constant — no
hardcoded paths remain in the worker bodies.

| Dataset | Version | Path | Constant(s) |
|---|---|---|---|
| FHS population totals (admin × year × age × sex) | **GBD 2023** (`release_id = 16`, FHS `location_set_id = 39`) | `/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2023_all_years.parquet` | `POP_TOTALS_PATH` |
| Gridded population raster (100 m and 1 km) | **2026_05_16** | `/mnt/team/rapidresponse/pub/population-model/results/2026_05_16/world_cylindrical_<meters>/<year>q1.tif` | `GRIDED_POP_PATH` |
| LBD admin shapefile release (regenerated locally to 0-360 and antimeridian-normalized parquets — see pre-stage sections) | **`2024_07_29`** | `/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29/` | `SHP_ROOT_NORMALIZED` (stage 3, 4B), `SHP_PATH_NORMALIZED_A0` + `SHP_ROOT_NORMALIZED_HIGHER` (stage 4A) |

To bump a version: edit the relevant constant at the top of each main
script in `03_admin_level_paf_main.py`,
`04_admin_level_exposure_a_main.py`, and `04_admin_level_exposure_b_main.py`
(and the matching files in `/ibtracs/`). For shapefile bumps, also
re-run the 0-360 and antimeridian-normalization pre-stages so the
team-mount parquets reflect the new release.

Note: the `/ibtracs/` folder mirrors most — but not all — of the main
scripts. The two Stage 4 downstream-task scripts
(`04_admin_level_exposure_a_resource_assignment.py` and
`04_admin_level_exposure_b_post_processing.py`) are **main-only**; the
ibtracs side has no matching files for them. See **Stage 4 — ibtracs
differences** below.

## Pre-stage — generating the FHS population totals parquet

`POP_TOTALS_PATH` points to `fhs_population_2023_all_years.parquet`, a
**concatenation of the GBD release's actuals (1970..latest GBD-actual
year) and the FHS forecast (post-GBD-actuals through 2100)**. The
forecast NetCDF carries values for the GBD-actual years too, so the
script drops `year_id ∈ {2023, 2024}` from the FHS side before concat to
avoid duplication ("we have actual population data for those years" —
see `population.py`). The hand-off year follows whatever the
configured GBD release reports as actuals (today: `release_id = 16`
publishes actuals through 2024). The file is built once per GBD/FHS
release by `population.py`.

### Script

`population.py`

### Inputs

| Source | Notes |
|---|---|
| GBD 2023 `get_population` via `db_queries` | `release_id = 16`, `age_group_id = 22` (all ages), `sex_id = 3` (both sexes), `year_id = 1970..2100`, location IDs from FHS hierarchy at `level <= 3` |
| FHS forecast NetCDF | `/mnt/share/forecasting/data/32/future/population/future_population_s130v41/population_agg.nc` — sliced to `sex_id = 3`, `age_group_id = 22`, draws mean-reduced |

### Outputs

```
/mnt/team/rapidresponse/pub/tropical-storms/
  fhs_population_2023.parquet              # GBD past only (1970..2022, single-row per loc/year)
  fhs_population_2023_future.parquet       # FHS forecast only (2023..2100, draws-mean)
  fhs_population_2023_all_years.parquet    # concat of the two — POP_TOTALS_PATH target
```

The third file is the one stage 3 + 4 actually read. The other two are
intermediate checkpoints kept for diagnostic / re-aggregation use.

### How to run

`db_queries` is an IHME shared library that requires the GBD conda
environment. Activate it first, then run the script:

```shell
source /ihme/code/central_comp/miniconda/bin/activate gbd_env
python population.py
```

### When to re-run

Re-run whenever any of these change:
- GBD release version (e.g., 2023 → 2024) — bump `gbd_2023_release_id` in the script.
- FHS forecast version — bump the `future_population_s130v41` path.
- Year range or age/sex filter — edit the `get_population(...)` call.

After generating a new parquet, point the pipeline at it by updating
`POP_TOTALS_PATH` in all 6 stage 3 + 4 main scripts (3 climada + 3 ibtracs).

---

## Pre-stage — generating the 0-360 admin shapefiles

`global_WGS84_admin{0,1,2}.parquet` on the team mount are the admin
polygons every non-NA-basin call in stages 3 + 4 reads (via `GDF_PATH` in
stages 3 + 4B, `GDF_ROOT` in stage 4A). They contain the same polygons
as the LBD source `.shp` files but shifted from the EPSG:4326 [-180, 180]
convention to **[0, 360]** longitudes, matching CLIMADA's basin centroid
convention. The file is built once per LBD release by
`generate_0_360_shapefiles.py`.

Stage 1 reads a different file (`global_WGS84.parquet`, no `_admin{N}`
suffix) — the **pre-glob** snapshot, intentionally pinned because Stage 1
is not being rerun. See Stage 1's Inputs for context.

### Script

`generate_0_360_shapefiles.py`

### Inputs

| Path | Purpose |
|---|---|
| `/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29/lbd_standard_admin_<N>.shp` | LBD admin polygons in [-180, 180]; N ∈ {0, 1, 2} |

### Outputs

```
/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/
  global_WGS84_admin0_0_360.parquet     # default; rename or re-target to overwrite canonical
  global_WGS84_admin1_0_360.parquet
  global_WGS84_admin2_0_360.parquet
```

The script writes to a `_0_360` suffix by default so the canonical
`global_WGS84_admin{N}.parquet` files (read by stages 3 + 4) aren't
clobbered until you've verified the regen. See **How to run** for how
to switch to the canonical names.

### Reprojection logic

`reproject_geometry` per polygon:
- Entirely in [0, 180] → unchanged.
- Entirely in [-180, 0] → shifted +360.
- Crosses the prime meridian normally → split at lon = 0, west half shifted +360.
- Crosses the antimeridian (touches ±180) → clipped at lon = 0, west half
  shifted, re-joined with a 1e-4 buffer to close the numerical seam.

### How to run

Sequential, no parallelism — sized for a one-off run after an LBD update.
Admin 2 is the slow one (~1 GB source).

```shell
conda run -n climada_env python generate_0_360_shapefiles.py
```

By default the script writes to `..._0_360.parquet` so existing canonical
files aren't clobbered. To replace `global_WGS84_admin{N}.parquet` directly,
set `OUTPUT_FILENAME_TEMPLATE` at the top of the script to
`"global_WGS84_admin{admin_level}.parquet"` before running.

### When to re-run

Re-run whenever:
- The LBD shapefile root bumps (currently `2024_07_29`) — edit `INPUT_ROOT`.
- A new admin level is needed downstream — add to `ADMIN_LEVELS`.

The output parquet paths are pinned at the top of every consumer
(`GDF_PATH` / `GDF_ROOT`) — no downstream edits needed after regenerating.

---

## Pre-stage — globbing special regions into the NA-basin admin 0 parquet

The NA-basin admin 0 shapefile read by stages 3 + 4
(`SHP_ROOT_NORMALIZED` in stages 3 + 4B, `SHP_PATH_NORMALIZED_A0` in
stage 4A) lives at `global_WGS84_admin0_normalized.parquet` on the team
mount. The `_normalized` suffix does two jobs:

1. **Antimeridian-normalized** in EPSG:4326 — inherited from the LBD
   source `.shp`, which is already in the -180..180 convention with
   antimeridian crossings split cleanly.
2. **Glob-merged** — 47 special-region / dependency `loc_id`s without
   FHS population coverage (Aruba, Cayman Islands, Saint-Barthélemy,
   Åland, …) are dissolved into the geometry of their administering
   admin 0 country (United Kingdom, Netherlands, France, …) via
   `shapely.ops.unary_union`. The child rows are dropped from the
   output; the parent rows carry the unioned geometry.

5 locations are intentionally left **standalone** (no glob, no drop):
Antarctica (60921), Liechtenstein (360), Vatican City (353), Paracel
Islands (60931), Spratly Islands (93924) — either sovereign states not
affected by tropical storms, or no natural parent admin 0.

### Script

`fix_missing_location_shapes.py`

### Inputs

| Path | Purpose |
|---|---|
| `/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29/lbd_standard_admin_0.shp` | LBD admin 0 polygons (already antimeridian-normalized in EPSG:4326); set via `SHP_ROOT_NORMALIZED` + `INPUT_PATH` |

### Outputs

```
/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/
  global_WGS84_admin0_normalized.parquet
```

Row count: input rows minus the 47 globbed children. The script will
fail fast if any of the 12 parent admin 0 `loc_id`s is missing from the
input (raises `ValueError`), and will warn (but not fail) if any child
`loc_id` is missing.

### Glob mapping

47 child `loc_id`s → 12 admin 0 parents. Source of truth:
`SPECIAL_TO_PARENT` dict at the top of the script.

| Parent (loc_id) | Child loc_ids |
|---|---|
| United Kingdom (95) | 296, 299, 313, 331, 345, 352, 355, 356, 368, 382, 415, 421, 60925, 60926, 60927 |
| France (80) | 338, 339, 350, 363, 364, 372, 387, 391, 394, 395, 423, 60348, 60930 |
| Netherlands (89) | 300, 4641, 4642, 60922 |
| Australia (71) | 318, 319, 375, 60924, 94026, 94027 |
| Norway (90) | 411, 60923 |
| Finland (79) | 297 (Åland) |
| Spain (92) | 311 (Canary Islands) |
| Denmark (78) | 332 (Faroe Islands) |
| Morocco (148) | 424 (Western Sahara) |
| United States (102) | 60928 (US Minor Outlying Islands) |
| Serbia (53) | 359 (Kosovo) |
| Cyprus (77) | 53483 (Northern Cyprus) |

Standalone (kept as-is): 60921, 60931, 93924, 360, 353.

### How to run

Sequential, no parallelism — sized for a one-off run after an LBD
update or a change to the special-region mapping.

```shell
conda run -n climada_env python fix_missing_location_shapes.py
```

Output is written directly to the canonical
`global_WGS84_admin0_normalized.parquet` path. **This will overwrite
the previous file** — no `_unglobbed` backup is taken. The script
prints per-parent merge progress and a final row-count summary.

### When to re-run

Re-run whenever:
- The LBD shapefile root bumps (currently `2024_07_29`) — edit
  `SHP_ROOT_NORMALIZED` at the top of the script.
- A new special region needs globbing (e.g., a future FHS release drops
  a current admin 0) — add the (child → parent) row to
  `SPECIAL_TO_PARENT`.
- An existing glob assignment changes (e.g., political reorganization,
  or a child should be moved to a different parent).

After regenerating, downstream consumers in stages 3 + 4 pick up the
new shapefile on next launch — no code edits needed. Stage 4A's
resource-estimation parquet (`resource_estimation_all_storms.parquet`)
should also be regenerated to drop tasks for the now-merged child
`loc_id`s; see Stage 4A's "Resource-assignment downstream task".

---

## Pre-stage — generating the Stage 3 task metadata parquet

`storm_draw_admin0_count.parquet` is the metadata file consumed by the
Stage 3 launcher (`03_admin_level_paf_launcher.py`) to enumerate all
tasks. It contains one row per
`(storm_draw, source_id, variant_label, experiment_id, batch_year, basin)`
combination (3 612 rows in the current run) and the per-task fields used
for task sizing and completion checking.

The Stage 3 launcher checks for this file at startup and **automatically
runs the generation script** if it is missing — no manual step is
required in normal circumstances. Re-run the script manually only when
the underlying data changes (see below).

### Script

`generate_storm_draw_admin0_count.py`

### Inputs

| Path | Purpose |
|---|---|
| `/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv` | All (model, variant, scenario, time_period, basin) combos to process |
| `/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv` | Maps storm draws to (source_id, variant_label) and sample columns |
| `stage1_v2/<source_id>/<variant_label>/<experiment_id>/<batch_year>/<basin>/intensity/*.zarr` | Stage 1 zarr files; `storm_XXXX/` subdirectory count = `n_storms_in_batch` |
| `stage2_v2/<storm_draw>/…/<year>/<basin>/raw_paf/*.tif` | Stage 2 PAF raster for the first year; used to count intersecting admin0 polygons |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0.parquet` | Admin0 polygon shapefile |

### Output columns

| Column | Description |
|---|---|
| `storm_draw` | e.g. `storm_draw_0002` |
| `source_id`, `variant_label`, `experiment_id`, `batch_year`, `basin` | Task key |
| `direct_rr_draw`, `indirect_cvd_draw`, `indirect_resp_draw` | Sample names from the storm draw table |
| `n_storms_in_batch` | Count of `storm_XXXX/` dirs inside the Stage 1 zarr |
| `estimated_storms_per_year` | `n_storms_in_batch / num_years_in_batch` |
| `year` | First year of the batch (string) |
| `num_admin0_first_year` | Unique admin0 regions intersecting the first-year PAF raster |
| `num_years_in_batch` | Number of years in the batch period |
| `estimated_admin0_total` | `num_admin0_first_year × num_years_in_batch` |

### Output path

```
/mnt/team/rapidresponse/pub/tropical-storms/climada/output/storm_draw_admin0_count.parquet
```

### How to run

```shell
python generate_storm_draw_admin0_count.py
```

Uses 64 cores via `rra_tools.parallel.run_parallel`. Expect ~30–60 minutes
depending on filesystem load. File permissions are set to `0o775` on write.

### When to re-run

Re-run when any of these change:
- Stage 1 zarr data is regenerated (changes `n_storms_in_batch`).
- Stage 2 PAF rasters are regenerated (changes `num_admin0_first_year`).
- The set of task combinations changes (new models, scenarios, basins, or
  storm draws added to `level_4_task_assignments.csv` or `storm_draw_table.csv`).

---

## Stage 1 — Per-storm intensity, exposure, and landfall locations

### Purpose

For every synthetic tropical cyclone track in a CMIP6 ensemble member, compute
three per-storm outputs at 0.1° resolution within the storm's basin:

1. **Intensity** — max wind speed per land pixel during the storm lifetime.
2. **Exposure hours** — per-pixel count of hours where wind speed exceeded
   the 17 m/s threshold, aggregated yearly.
3. **Landfall locations** — per-administrative-area (admin0) max wind speed,
   one row per qualifying location.

Only storms that produce ≥ 17 m/s winds on land receive intensity and
exposure outputs. Every storm receives a landfall parquet (possibly empty)
that serves as the completion sentinel for resume logic.

### Script
`01_climada_intensity_main.py`

### Runtime scope

Unique draw-runs per launcher submission =
`N_combinations × 7 basins × 100 draws`, where `N_combinations` is the
row count of `(source_id, variant_label, experiment_id, batch_year)` in
the active CMIP6 task assignment file.

### CLI

```
python 01_climada_intensity_main.py \
  --source_id <str> \
  --variant_label <str> \
  --experiment_id <str> \
  --batch_year <YYYY-YYYY> \
  --basin <EP|NA|NI|SI|AU|SP|WP> \
  --draw_batch <start-end>   # e.g. "0-9" runs draws 0..9 inclusive
  --num_cores <int>          # default 1; parallelism is over draws
```

### Inputs

| Path | Purpose |
|---|---|
| `/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/<source_id>/<variant_label>/<experiment_id>/<batch_year>/<basin>/tracks_*.nc` | Synthetic TC tracks (one NetCDF per draw; multi-storm along `n_trk`) |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84.parquet` | Land polygons (0-360 lon) for all basins except NA. This is the **pre-glob** snapshot from before the special-region merge; intentionally pinned for Stage 1 because Stage 1 is not being rerun. Stages 3 + 4 read the post-glob `global_WGS84_admin0.parquet` produced by `generate_0_360_shapefiles.py` — divergence between this file and `_admin0.parquet` is expected. |
| `/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29/lbd_standard_admin_0.shp` | Land polygons (−180..180 lon, normalized) for the NA basin |

Both gdfs share columns `loc_id` and `ADM0_NAME` used downstream.

### Outputs

All outputs live under
`/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1/<source_id>/<variant_label>/<experiment_id>/<batch_year>/<basin>/`.

| Output | Path suffix |
|---|---|
| Intensity zarr | `intensity/intensity_<basin>_<source>_<exp>_<variant>_<start>01_<end>12<draw_text>.zarr/storm_NNNN/` |
| Exposure zarr  | `exposure_hours/exposure_hours_<basin>_<source>_<exp>_<variant>_<start>01_<end>12<draw_text>.zarr/storm_NNNN/` |
| Landfall parquet | `landfall_locations/landfall_<basin>_<source>_<exp>_<variant>_<start>01_<end>12<draw_text>/storm_NNNN.parquet` |
| Draw completion marker | `<LOG_DIR>/draw_completion_markers/<source>/<variant>/<experiment>/<batch_year>/<basin>/draw_NNNN.json` |

Conventions:
- `<draw_text>` is empty for draw 0; otherwise `_e{draw-1}` (CMIP ensemble member suffix).
- Zarr group key per storm is `storm_{storm_index:04d}` (4-digit zero-padded).
- Zarr v3 with blosc/zstd level 9 + bitshuffle, chunked `(lat=64, lon=64)`, fp32.
- Landfall parquet columns: `source_id, variant_label, experiment_id, batch_year, basin, draw, storm_id, start_date, loc_id, ADM0_NAME, max_wind_m_s`.

### Constants

| Constant | Value | Notes |
|---|---|---|
| `RESOLUTION` | `0.1` degrees | Centroid grid spacing |
| Centroid `buffer_deg` | `5.0` | Added to each basin's lon/lat bounds |
| Land subset `buffer` | `0.25` degrees | Pad applied to storm bbox before spatial join |
| Wind threshold | `17.0` m/s | Used by landfall check, exposure, and per-loc max filter |

### Basin bounds (0-360 lon convention)

| Basin | lon_min | lon_max | lat_min | lat_max |
|---|---|---|---|---|
| EP | 180 | 290 | 0 | 60 |
| NA | 260 | 360 | 0 | 60 |
| NI | 30 | 100 | 0 | 50 |
| SI | 20 | 100 | -45 | 0 |
| AU | 100 | 180 | -45 | 0 |
| SP | 180 | 250 | -45 | 0 |
| WP | 100 | 180 | 0 | 60 |

### Longitude conventions

- CLIMADA centroids stay in **0-360** to match IBTrACS basin definitions.
- Intensity and exposure outputs are **normalized to −180..180 for NA only**
  after CLIMADA emits them (via `normalize_lon_to_180`). This matches the
  normalized NA shapefile used for land clipping.
- All other basins remain in 0-360 end-to-end.

### Pipeline flow

Per-draw (`process_single_draw`):

1. Skip if the draw's completion marker exists.
2. Clean up any zarr stores with `.partial` files (orphans from prior crashes).
3. Open the draw's NetCDF.
4. Load the basin-appropriate global land gdf **once** and force-build its
   spatial index (`load_global_land_gdf`).
5. Compute basin centroids once (`generate_basin_centroids`).
6. Vectorized scan for valid storm indices (`get_storm_indices`).
7. Iterate storms sequentially, each wrapped in `try/except` so one failure
   doesn't kill the draw.
8. After the loop, chmod the written zarr stores and write the draw
   completion marker.

Per-storm (`process_single_storm`):

1. **Resume check** — `check_existing_storm_in_zarr` returns True if the
   landfall parquet exists and (if non-empty) the zarrs also validate.
2. Slice the storm from the open NetCDF
   (`read_single_storm_from_dataset` — trims to first/last ANY-finite step).
3. Normalize to CLIMADA's track schema (`normalize_nc_storm_for_climada`).
4. `TropCyclone.from_tracks(... store_windfields=True)` — heaviest step.
5. Compute intensity grid (`generate_intensity_per_storm`).
6. For NA: `normalize_lon_to_180(storm_intensity)`.
7. Spatial-index lookup for the storm's land polygons
   (`subset_land_for_storm` — uses `gdf.sindex.query` with `intersects`).
8. Storm-level landfall check (`check_storm_landfall` — fast union geom).
9. Clip intensity to land → save intensity zarr.
10. Per-polygon zonal max → save landfall parquet.
11. Compute wind-speed cube → yearly exposure hours
    (`compute_yearly_exposure_per_storm`).
12. Clip exposure to land → save exposure zarr.

### Case handling

| Case | Condition | Intensity zarr | Exposure zarr | Landfall parquet |
|---|---|---|---|---|
| **0 Resume** | Sentinel parquet says "done" | (skipped) | (skipped) | (already on disk) |
| **1 NoIntensity** | CLIMADA produced zero wind at every centroid | — | — | empty |
| **2 No land in bbox** | Storm's buffered bbox doesn't intersect any polygon | — | — | empty |
| **3 No landfall** | No wind > 17 m/s OR no strong winds reach land | — | — | empty |
| **4 Landfall** | `check_storm_landfall` returns True; ≥1 polygon clears 17 m/s | written | written | rows |
| **4b Edge** | Storm-level passes but no single polygon clears 17 m/s | written | written | empty |
| **5 Error** | Any other exception in `process_single_storm` | — | — | — (retry next run) |

### Resume semantics

The **landfall parquet is the per-storm completion sentinel**:

- **Missing** → storm not yet processed; full pipeline runs.
- **Empty** → storm was processed and found to make no qualifying landfall;
  no zarrs are expected. Storm short-circuits on subsequent runs.
- **Non-empty** → storm makes landfall; zarrs must also exist and pass
  structural validation (`check_existing_storm_in_zarr`). If a zarr is
  missing or has a `.partial` file, that storm's zarr group is deleted and
  the storm is reprocessed.

Draw-level completion is marked separately by a JSON file in
`<LOG_DIR>/draw_completion_markers/...`. A draw is skipped at the top of
`process_single_draw` if its marker exists. Per-storm sentinels handle
partial-draw recovery; the draw marker handles "skip the whole draw".

#### Launcher-level resume (batch granularity)

The launcher (`01_climada_intensity_launcher.py`) submits **draw-batches**
(e.g., `"0-4"` = draws 0..4). It derives completion by scanning
`<LOG_DIR>/draw_completion_markers/...` for per-draw markers and treats a
batch as done iff every draw in it has a marker.

If a batch crashes part-way (e.g., 3 of 5 draws marked), the launcher
**reruns the whole batch**. The cost of re-submitting a partially-done batch
is bounded: inside the rerun, the main script's per-storm
`check_existing_storm_in_zarr` short-circuits every already-done storm in
the already-done draws, so the only real work is the catch-up draws plus a
small per-storm stat check on what's already on disk.

### Performance characteristics

- **Land gdf loaded once per draw**, not once per storm. The spatial index
  is built eagerly so the first per-storm bbox query doesn't pay for it.
- **Unioned land geometry computed once per storm** and reused for the
  landfall check and both `clip_raster_to_land` calls (intensity + exposure).
- **`get_storm_indices` and `read_single_storm_from_dataset` are vectorized**
  across `n_trk` and `time`; no Python loops over storms or timesteps.
- **No-landfall storms short-circuit** before re-running CLIMADA on resume
  (sentinel parquet existence is enough).
- Each worker process loads the full land gdf independently — for
  whole-machine parallelism over draws this is multiplied by `num_cores`.

### Failure modes worth knowing

- **CLIMADA grid-mismatch** (`ValueError` from `generate_intensity_per_storm`
  or `generate_speed_per_storm`) — propagates to the storm-level try/except,
  logs full traceback, no sentinel written, storm reprocessed on next run.
- **`.partial` files** — orphans from a killed Python process mid-write.
  `check_and_cleanup_zarr_store` deletes such zarr stores at the start of
  each draw; `check_existing_storm_in_zarr` deletes individual storm groups
  if it finds a `.partial` inside.
- **NA shapefile read latency** — `lbd_standard_admin_0.shp` lives on
  `/snfs1` and is slower to read than the team-mount parquet. Stages 3
  and 4 already migrated to the antimeridian-normalized parquet at
  `global_WGS84_admin0_normalized.parquet`; Stage 1 still reads the
  `.shp` directly. Migrating Stage 1 to read the same parquet would
  shave a few seconds off every draw's startup.

---

## Stage 2 — Per-storm relative risk → yearly draw-mean PAF

### Purpose

Consume stage 1 per-storm intensity zarrs and produce, per year, a basin-wide
**Population Attributable Fraction (PAF)** raster averaged across all 100
CLIMADA inner draws of a given storm_draw / scenario.

For each storm in each inner draw:

1. Look up per-pixel **relative risk (RR)** from windspeed using an empirical
   RR curve (one of `indirect_resp_draw` or `indirect_cvd_draw`), drawing a
   single column (`sample_name`) from the table.
2. Convert per-storm RR into a per-pixel PAF contribution using
   `PAF = (RR − 1) / RR × (days / 365)`, where `days = 20` is a fixed impact
   window applied to every pixel inside the storm footprint.
3. Sum PAF contributions across all storms in a year for one inner draw.
4. After all 100 inner draws are done, average to produce the **draw-mean PAF**
   per year — that's what gets saved.

### Script

`02_relative_risk_main.py`

### Runtime scope

One invocation processes a single tuple of
`(storm_draw, source_id, variant_label, experiment_id, batch_year, basin,
relative_risk, sample_name)`. Inside that invocation:

- 100 inner CLIMADA draws (range `0..99`) are processed by
  `process_single_draw` in **batches of `num_cores`** via
  `rra_tools.parallel.run_parallel`.
- All storms in a draw × all years in `batch_year` are accumulated into a
  per-year `cumulative_paf[year]` and divided by 100 at the end.
- One GeoTIFF is written per year in `batch_year`.

### CLI

```
python 02_relative_risk_main.py \
  --storm_draw <storm_NNNN>           # outer-level identifier; path component only
  --source_id <str> \
  --variant_label <str> \
  --experiment_id <str> \
  --batch_year <YYYY-YYYY> \
  --basin <EP|NA|NI|SI|AU|SP|WP> \
  --relative_risk <indirect_resp_draw|indirect_cvd_draw> \
  --sample_name <str>                 # column name in the RR table
  --num_cores <int>
```

### Inputs

| Path | Purpose |
|---|---|
| `/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1/<source>/<variant>/<experiment>/<batch_year>/<basin>/intensity/intensity_*.zarr/storm_NNNN/` | Stage 1 per-storm intensity rasters (one zarr group per storm, one zarr store per inner draw) |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/relative_risk_samples/rd_rr_samples.csv` | Respiratory-disease RR curve (`relative_risk=indirect_resp_draw`) |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/relative_risk_samples/cvd_rr_samples.csv` | Cardiovascular-disease RR curve (`relative_risk=indirect_cvd_draw`) |

Both RR CSVs share columns `windspeed` (knots) and per-sample RR columns
(e.g., `mean`, `sample_001`, …); one column at a time is selected via
`--sample_name`.

### Outputs

`SAVE_ROOT = /mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage2/`

One GeoTIFF per year:
```
<SAVE_ROOT>/<storm_draw>/<source>/<variant>/<experiment>/<batch_year>/<year>/<basin>/raw_paf/
  draw_mean_raw_paf_<storm_draw>_<relative_risk>_<sample>_<basin>_<source>_<exp>_<variant>_<start>01_<end>12_<year>.tif
```

Conventions:
- One file per year in `batch_year` (inclusive on both ends).
- Float32, GTiff compressed with deflate + predictor 3, tiled `256×256`.
- Saved values are the **mean across all 100 inner draws** of the per-year
  per-pixel raw PAF.
- File mode `0o664`; parent dir `0o775` (set after each save).

### Constants

| Constant | Value | Notes |
|---|---|---|
| `IMPACT_DAYS_FRACTION` | `20.0 / 365.0` | Per-pixel year-fraction applied wherever RR > 0. Previously materialized as a per-storm `days_impact` raster of constant 20 — collapsed because the mask reduces to `rr > 0` and the value is constant. |
| Template raster `res` | `0.1` degrees | Basin-wide grid resolution; matches stage 1 |
| Template raster `buffer_deg` | `5.0` | Added to each basin's lon/lat bounds (matches stage 1) |
| Inner draws | `100` | Hardcoded as `draws = list(range(100))` in `main` |

### PAF formula

For each storm-affected pixel `p` in inner draw `d` and year `y`:

```
paf_contribution[d, y, p] = (RR[d, y, p] − 1) / RR[d, y, p] × IMPACT_DAYS_FRACTION
```

Summed across storms in the same `(d, y)`, then averaged across draws:

```
raw_paf[y, p] = mean_{d=0..99}(  sum_{storms in (d, y)} paf_contribution  )
```

Note: PAF contributions are signed — `RR < 1` produces negative pixels, and
multiple storms hitting the same pixel can sum to > 1. No clamping is applied;
downstream stages may renormalize.

### Pipeline flow

Per-invocation (`main`):

1. **Resume check (whole draw)** — `check_if_draw_complete` returns True
   iff every year's GeoTIFF exists, is ≥ 1 KB, and loads as a valid raster.
   If True, return.
2. **Per-year resume** — `get_year_status` partitions years into
   `valid_years` (file passes the same checks) and `invalid_years`. Only
   `invalid_years` are recomputed. Invalid files are unlinked.
3. Build the basin template raster once (`generate_basin_template_raster`).
4. Load the RR sample CSV once.
5. Initialize `cumulative_paf = {year: zeros(template_shape)}` for each
   invalid year.
6. Iterate inner draws `0..99` in batches of `num_cores`, calling
   `process_single_draw` via `run_parallel`. Sum each worker's returned
   `yearly_paf[year]` into `cumulative_paf[year]`.
7. Divide by `n_draws = 100` to get the draw-mean.
8. Save one GeoTIFF per invalid year, collect the saved paths.
9. Single chmod sweep over `(file, file.parent)` for every saved path.

Per inner draw, per worker (`process_single_draw`):

1. Build the per-draw `RRInterpolator` from the shipped `rr_samples_df` +
   `sample_name` (`build_rr_interpolator`). Reused for every storm.
2. Locate the stage 1 intensity zarr (`get_draw_zarr_path`). If missing or
   empty → return zero rasters for all invalid years.
3. Read all storm metadata once (`iter_storms_metadata`) to map storms to
   the years they affect (`map_storms_to_years`).
4. For each year:
   a. If the year has no storms → fill with zeros and continue.
   b. For each storm in that year (try/except per storm; on failure log
      traceback and `continue`):
      - Open the storm zarr in a `with` block (handle released on any exit).
      - `ensure_min_grid` to guarantee ≥ 3×3 pixels for rasterization.
      - For NA basin: `normalize_dataset` (0-360 → −180..180).
      - Compute RR via `generate_relative_risk` + the pre-built interpolator.
      - Mask non-positive RR to NaN (`where(rr_da > 0)`).
      - Convert to `rt.RasterArray`, resample (nearest) to the basin template.
      - Mask `= rr_values > 0` (NumPy `>` excludes NaN). If any: accumulate
        `(rr − 1)/rr × IMPACT_DAYS_FRACTION` into `sum_raw_paf`.

### Resume semantics

The completion sentinel is **per-year GeoTIFF existence + validity**:

- **Missing** → year not yet computed; included in `invalid_years`.
- **Size < 1 KB** → corrupt; file is unlinked, year goes to `invalid_years`.
- **Fails `rt.load_raster`** → unreadable; file is unlinked, year goes to
  `invalid_years`.
- **All years valid** → script returns immediately at the top of `main` via
  `check_if_draw_complete`.

If a single year is corrupt, only that year is recomputed (all 100 inner
draws are re-run *just to reconstruct that year's PAF surface*). There is
no per-(inner draw) sentinel — the inner-draw accumulator runs in memory
and is only persisted as a mean.

#### Launcher-level resume (task granularity)

The launcher (`02_relative_risk_launcher.py`) submits one Jobmon task per
8-tuple `(storm_draw, source_id, variant_label, experiment_id, batch_year,
basin, relative_risk, sample_name)`. It derives completion by **scanning the
filesystem under `SAVE_ROOT`** rather than querying Jobmon.

Per-row check (`task_is_complete`):
1. For each `year` in `batch_year`, build the expected raw_paf GeoTIFF path
   via `_stage2_paf_path` — must match the format `save_raster` writes.
2. Treat the task as complete iff every year's TIF exists *and* is ≥ 1 KB.
3. Skip the load-validity step at launcher level — the main script's
   `_is_valid_raster` will catch and rebuild corrupt files once the task
   actually runs.

If any year is missing or undersized for an 8-tuple, the launcher submits a
task for it. Inside that task, `get_year_status` decides per-year which to
recompute, so submitting a partially-done task is cheap (the main script
short-circuits already-valid years).

Submission gating (`PRIORITY_MODE` constant in `02_relative_risk_launcher.py`):
- `"non_priority"` — submit everything except the hardcoded `PRIORITY_DRAWS`
  list of 8 storm_draws.
- `"priority"` — submit only `PRIORITY_DRAWS` (smoke runs).
- `"all"` (current value) — submit every storm_draw, priority ones first.

Note: the code comment block labels `"non_priority"` as the default, but the
runtime constant is currently set to `"all"`. Treat the comment as
historical and the constant as authoritative.

Rerun runtime: `remaining_long["max_run_time"] *= 3` after the completion
scan. The per-task runtime in `stage2_resource_usage.parquet` reflects
first-time success; reruns can be slower because corrupt single-year
recomputes still loop over all 100 inner draws inside `main`.

### Performance characteristics

- **`interp1d` built once per worker (per inner draw)** in
  `build_rr_interpolator`, not once per storm. The `RRInterpolator`
  (`NamedTuple`) carries the precomputed callable, windspeed bounds, and
  upper-cap RR value for all storms in the draw.
- **No per-storm `days_impact` raster.** The mask
  `np.isfinite(t_impact) & np.isfinite(rr_values) & (t_impact > 0) & (rr_values != 0)`
  collapses to `rr_values > 0` and the value is constant
  (`IMPACT_DAYS_FRACTION`), so one full raster construction + resample per
  storm is eliminated.
- **`interpolate_rr_from_windspeed` constructs a fresh DataArray** rather
  than `.copy()` + `.values =` overwrite — no wasted allocation of the
  source intensity values.
- **Per-storm try/except** + `with xr.open_zarr(...) as storm_ds:` so one
  bad storm logs traceback and the loop continues; file handles always
  released.
- **Tight chmod** — after `save_raster` returns the saved path, a single
  loop applies `os.chmod(path, 0o664)` + `os.chmod(path.parent, 0o775)`.
  No `os.walk` tree traversal.

### Failure modes worth knowing

- **Single corrupt year on rerun** — `get_year_status` unlinks the bad file
  and recomputes only that year. All 100 inner draws still re-run for that
  year (the inner accumulator is in-memory only). If you want to avoid
  the full recompute, manually copy a valid neighbor year's file or accept
  the cost.
- **Stage 1 zarr missing or empty** — `process_single_draw` returns zero
  rasters for that inner draw rather than failing. The draw-mean for that
  year is silently biased toward zero by missing input.
- **`storm_draw` is path-only** — currently the CLI `storm_draw` arg
  appears only in output paths, not in which stage 1 zarrs are read. If
  upstream needs per-`storm_draw` input selection, that wiring is not
  present today.
- **Single `sample_name` per invocation** — to produce per-draw uncertainty
  outputs (`sample_001`, …, `sample_100`), run the script 100 times with
  different `--sample_name` values. The 100× workflow is launcher-level,
  not main-script-level.

---

## Stage 3 — Per-admin population-weighted PAF

### Purpose

Take stage 2's basin-wide raw PAF rasters and produce, per year, a parquet
of **population-weighted PAF per admin0 location**. For each admin polygon
that intersects the basin's affected area:

1. Clip the PAF raster to the admin polygon.
2. Load the gridded population raster (100 m, equal-area) bounded to the
   admin's reprojected bbox.
3. Resample PAF to the population grid.
4. Compute the storm-exposed numerator: `Σ(pop × paf)` over pixels where
   both are positive.
5. Divide by the admin's **full** population (from the FHS totals parquet —
   *not* just the exposed pixels) to get the population-weighted PAF.

The final value is interpretable as "fraction of admin-level person-time
attributable to TC-windspeed exposure in this year."

### Script

`03_admin_level_paf_main.py`

### Runtime scope

Same 8-tuple per invocation as stage 2:
`(storm_draw, source_id, variant_label, experiment_id, batch_year, basin,
relative_risk, sample_name)`. Inside:

- One year in `batch_year` per parallel worker via `run_parallel`, batched by
  `num_cores`.
- Each worker computes population-weighted PAF for every admin polygon that
  intersects the basin's RR > 0 area for that year.

### CLI

```
python 03_admin_level_paf_main.py \
  --storm_draw <storm_NNNN> \
  --source_id <str> \
  --variant_label <str> \
  --experiment_id <str> \
  --batch_year <YYYY-YYYY> \
  --basin <EP|NA|NI|SI|AU|SP|WP> \
  --relative_risk <indirect_resp_draw|indirect_cvd_draw> \
  --sample_name <str> \
  --num_cores <int>
```

### Inputs

| Path | Purpose |
|---|---|
| `<stage2>/<storm_draw>/.../<year>/<basin>/raw_paf/draw_mean_raw_paf_*.tif` | Stage 2 per-year PAF rasters (one per year in batch_year) |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0.parquet` | Admin0 polygons for all basins except NA (WGS84, -180..180 lon) |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0_normalized.parquet` | Admin0 polygons for the NA basin (antimeridian-normalized to -180..180, with special regions globbed into their admin 0 parents). Generated by `fix_missing_location_shapes.py` — see **Pre-stage — globbing special regions into the NA-basin admin 0 parquet**. Set via `SHP_ROOT_NORMALIZED`. |
| `/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2023_all_years.parquet` | FHS population totals by `location_id × year_id × age_group_id × sex_id` (set via `POP_TOTALS_PATH`) |
| `/mnt/team/rapidresponse/pub/population-model/results/2026_05_16/world_cylindrical_100/<year>q1.tif` | Gridded population, 100 m, ESRI:54034 (cylindrical equal-area) (set via `GRIDED_POP_PATH`) |

The FHS query filters to `age_group_id = 22` (all ages) and `sex_id = 3`
(both sexes). Missing admin × year returns a sentinel population of `1.0`
with `special_region_flag = True` so the row is preserved with a clearly
flagged denominator.

### Outputs

`SAVE_ROOT = /mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage3/`

One parquet per year:
```
<SAVE_ROOT>/<storm_draw>/<source>/<variant>/<exp>/<batch_year>/<basin>/<year>/paf_df/
  paf_<storm_draw>_<rr>_<sample>_<basin>_<source>_<exp>_<variant>_<start>01_<end>12_<year>.parquet
```

Per-row columns:
| Column | Notes |
|---|---|
| `storm_draw` | str, e.g. `storm_draw_0042` |
| `location_id` | admin0 `loc_id` from the shapefile |
| `year` | int, the four-digit year |
| `total_population` | FHS admin total for that year (sentinel `1.0` if missing) |
| `population_exposed` | Σ population within RR > 0 pixels (NOT the denominator) |
| `population_weighted_paf` | Σ(pop × paf) / `total_population` |
| `relative_risk` | the CLI `--relative_risk` value |
| `special_region_flag` | `True` if `total_population` came from the missing-row sentinel |

File mode `0o775`; parent dir `0o775` (set inside `save_batch_paf_dataframe`,
so each worker is self-contained for permissions).

### Constants

| Constant | Value | Notes |
|---|---|---|
| `ANTIMERIDIAN` | `LineString([(180, -90), (180, 90)])` | Used by `split_antimeridian_geom` for polygons that wrap |
| Per-piece bbox buffer | `0.2°` | Pad applied to each piece's CEA bbox before loading population |
| Population raster resolution | `100` m | `load_in_gridded_population` accepts `meters` as a parameter; `process_single_year` always passes `100` |
| Equal-area CRS | `ESRI:54034` | Cylindrical equal area; matches the gridded population's projection |

### Pipeline flow

Per-invocation (`main`):
1. Enumerate years from `batch_year` (inclusive on both ends).
2. Build one args tuple per year and dispatch to `process_single_year` via
   `run_parallel(num_cores=num_cores)`.
3. Each worker writes its parquet + chmods it; no post-parallel sweep needed.

Per-year, per-worker (`process_single_year`):
1. **Resume check** — `check_if_year_complete` returns True iff the parquet
   exists, is > 0 bytes, opens via `pyarrow.parquet.ParquetFile`, and has
   ≥ 1 row. If True, return.
2. Load the basin-appropriate admin0 shapefile + FHS population totals.
3. Load the stage 2 PAF raster + clean it (0 → NaN, crop to affected area
   via `subset_affected_area`).
4. `intersect_shapefile_with_raster` — find admin polygons whose footprint
   intersects RR > 0 pixels in the cleaned raster.
5. `reproject_shapefile_to_equal_area` — if the gdf crosses the
   antimeridian (`maxx > 180`), normalize longitudes; reproject the
   intersected shapes to `ESRI:54034`; polygonize. No CEA-bbox clipping
   is done at this stage — stage 4 has a separate corner-transform
   helper for that.
6. For each admin polygon (try/except scoped — one bad polygon doesn't kill
   the year):
   a. `split_antimeridian_geom` — produce one or two (CEA, WGS84) pairs
      depending on whether the polygon wraps.
   b. For each piece:
      - For non-NA basins: shift the WGS84 piece to 0-360 to match the
        stage 2 PAF raster's lon convention.
      - Clip + mask the PAF raster to the WGS84 piece.
      - Compute the piece's CEA-bounded bbox; load gridded population
        bounded by that bbox.
      - Mask population by the CEA piece; on `WindowError` (no overlap),
        skip the piece and continue.
      - Resample PAF to the population grid (nearest).
      - Build `valid_mask = (pop > 0) & isfinite(pop) & (paf > 0) & isfinite(paf)`.
      - Accumulate `numerator += Σ(pop[mask] × paf[mask])` and
        `population_exposed_total += Σ(pop[mask])`.
   c. `get_population_total(pop_df, year, admin_id)` — admin's full FHS
      population. Returns `(1.0, True)` if no row found.
   d. `population_weighted_paf = numerator / total_population` (zero if the
      denominator is zero).
   e. Append a record to `paf_records`.
7. Concat records into a sorted DataFrame and save to parquet (chmod inside
   the save function).

### Resume semantics

Per-year parquet is the sentinel. `check_if_year_complete` requires:
- file exists,
- size > 0 bytes,
- opens cleanly via `pyarrow.parquet.ParquetFile`,
- has ≥ 1 data row.

Any failure → that year goes into the worker queue. Already-valid years are
skipped at the top of `process_single_year`. There is no per-admin sentinel
— if a year is rerun, every admin polygon recomputes.

Per-admin failures are isolated via a `try/except` inside the admin loop:
the failing polygon logs a traceback and the loop continues. The year still
saves a parquet, just with the failed admins absent. This means a corrupt
polygon won't infinitely re-queue the year, but does mean a downstream
consumer should treat missing admins for a known basin as "compute failed,
investigate" rather than "no exposure."

#### Launcher-level resume (task granularity)

The launcher (`03_admin_level_paf_launcher.py`) submits one Jobmon task per
8-tuple `(storm_draw, source_id, variant_label, experiment_id, batch_year,
basin, relative_risk, sample_name)`. It derives completion by **scanning the
filesystem under `SAVE_ROOT`** rather than querying Jobmon.

Per-row check (`task_is_complete`):
1. For each `year` in `batch_year`, build the expected paf parquet path via
   `_stage3_paf_path` — must match the format `save_batch_paf_dataframe`
   writes in the main script.
2. Treat the task as complete iff every year's parquet exists *and* is ≥ 1 KB.
3. Skip the parquet header / row-count validity test at launcher level —
   the main script's `check_if_year_complete` will catch and rebuild
   invalid files once the task actually runs.

If any year is missing or undersized for an 8-tuple, the launcher submits a
task for it. Inside that task, `check_if_year_complete` decides per-year
which to recompute, so submitting a partially-done task is cheap (already-
valid years short-circuit at the top of `process_single_year`).

**Bug fix on the way:** the previous Jobmon-based completion path used a
DataFrame merge that omitted `sample_name` from the `on=` keys. That meant
once any one sample for a 7-tuple completed, all other samples for the
same 7-tuple were silently marked done and dropped from submission. The
filesystem check keys on the full 8-tuple, so each sample is now evaluated
independently.

Submission gating:
- `PRIORITY_MODE = "non_priority"` (default) — submit everything except the
  hardcoded `PRIORITY_DRAWS` list of 8 storm_draws.
- `PRIORITY_MODE = "priority"` — submit only `PRIORITY_DRAWS` (smoke runs).
- `PRIORITY_MODE = "all"` — submit every storm_draw, priority ones first.

Resource budget: stage 3 uses fixed per-task constants (`DEFAULT_NUM_CORES=1`,
`DEFAULT_MAX_RUN_TIME_MIN=10`, `DEFAULT_MEMORY_GB=25`) — no resource-usage
parquet like stage 2. Reruns triple **both** runtime AND memory to give
headroom for slow-corrupt-year recomputes:
```python
remaining_long["max_run_time"] = remaining_long["max_run_time"] * 3
remaining_long["memory_gb"] = remaining_long["memory_gb"] * 3
```

### Performance characteristics

- **`itertuples(index=False)` over admin polygons** instead of `iterrows()`.
  For NA-basin years with ~50-100 admin polygons that intersect, the
  per-row Python overhead drops ~10×.
- **No explicit `gc.collect()` in hot loops.** CPython's refcount-based
  collector reclaims memory immediately on `del`; explicit cycle-collection
  was a no-op on linear pipelines like this one and was adding measurable
  per-iteration overhead.
- **Per-worker chmod** — `save_batch_paf_dataframe` sets file + parent dir
  permissions immediately after the parquet write. No `os.walk` tree walk
  across years in `main`.
- **Bounded population reads** — `load_in_gridded_population(bounds=...)`
  reads only the CEA-bounded window of the population tiff per piece.

### Failure modes worth knowing

- **Missing stage 2 PAF for a year** — `load_raw_paf_raster` raises
  `FileNotFoundError`. This propagates through `process_single_year` (no
  per-year try/except in `main`), so `run_parallel` records the year as
  failed but the other year workers continue. The launcher (or a rerun)
  should see the missing per-year parquet and resubmit.
- **WindowError on piece × population intersection** — handled inside the
  per-piece loop (a piece's CEA bbox might not overlap the population
  raster); the piece is skipped and the admin's numerator accumulator
  continues with whatever pieces did intersect.
- **Missing admin × year in FHS totals** — `get_population_total` returns
  the sentinel `(1.0, True)` and the row is saved with
  `special_region_flag=True`. The `population_weighted_paf` for those rows
  is effectively `numerator / 1.0`, which is the raw exposed-person-PAF
  sum, not a true population fraction. Downstream code should filter on
  `special_region_flag` before aggregating.
- **Lon-convention drift** — stage 2 PAF rasters for non-NA basins live in
  0-360 lon; admin shapes are in -180..180. The per-piece path explicitly
  shifts the WGS84 piece to 0-360 via `normalize_geom_to_0_360` for
  non-NA basins before clipping the raster. The CEA bbox helper does the
  reverse shift (subtract 360 if `xmin > 180`) so pyproj receives lons in
  the -180..180 range it expects.

## Stage 4 — Per-(storm, location) population-weighted exposure

Stage 4 has four pieces. Pipeline order is **Part A → resource-assignment
bridge → Part B → post-processing**.

- **Part A** (`04_admin_level_exposure_a_main.py`) — per-(storm, admin)
  metadata pass. For every storm × admin polygon that intersects, records
  pixel counts, percent affected, area, and bbox. Output drives Part B's
  resource budgeting.
- **Resource-assignment bridge**
  (`04_admin_level_exposure_a_resource_assignment.py`) — compiles every
  Part A `admin_level_metadata` parquet, bins each (storm × location_id)
  task into a size class based on `area_100m2`, and writes the parquet
  that Part B's launcher consumes. Runs as a downstream Jobmon task in
  the 4A workflow with all per-(combo × draw) tasks as upstream
  dependencies (see Part A's "Resource-assignment downstream task"
  subsection).
- **Part B** (`04_admin_level_exposure_b_main.py`) — the fine-grained
  worker that computes person-storm-hours and max wind speed for one
  `(storm × location_id)` and writes a per-row parquet.
- **Post-processing** (`04_admin_level_exposure_b_post_processing.py`)
  — walks the 35M+ expected 4B outputs (paths enumerated from 4A's
  `compiled_admin_level_metadata.parquet`, not from `rglob`) and concats
  them into one consolidated parquet at
  `stage4b/_consolidated/storm_exposure_all.parquet`. Runs as a
  downstream Jobmon task in the 4B workflow with all per-(storm × loc)
  tasks as upstream dependencies (see Part B's "Post-processing
  downstream task" subsection).

### Stage 4 — ibtracs differences

The two downstream-task scripts above
(`04_admin_level_exposure_a_resource_assignment.py` and
`04_admin_level_exposure_b_post_processing.py`) are **main-only**. They
exist to manage the synthetic-CMIP6 pipeline's scale (35M+ tasks) and
have no equivalent in `/ibtracs/`, which operates on the much smaller
set of historical observed storms. On the ibtracs side:

- **Resource assignment is inline, not a separate task.** Where main
  runs `..._a_resource_assignment.py` as a downstream Jobmon task to
  compile every Part A metadata parquet and bin each (storm × location)
  task into a size class by `area_100m2`, the ibtracs launchers assign
  **fixed** resource budgets directly. `04_admin_level_exposure_a_launcher.py`
  sets a flat `memory = '15G'`, and `04_admin_level_exposure_b_launcher.py`
  reads 4A's per-storm metadata parquets directly
  (`META_ROOT.glob("**/*.parquet")`) and sets a flat `memory = '5G'`.
  No compiled-metadata bridge parquet is produced or consumed.

- **There is no output-consolidation step.** The ibtracs 4B launcher
  adds only the per-(storm × location) compute tasks — there is no
  downstream post-processing task and no `stage4b/_consolidated/...`
  parquet. At ibtracs scale the per-storm outputs are consumed directly.

To bump data versions on the ibtracs side, edit the same module-level
constants in the ibtracs `*_main.py` files; there is nothing extra to
edit for the (absent) downstream tasks.

### Part B — Per-(storm, location) compute

#### Purpose

For one `(storm × location_id)`:
1. Load the storm's intensity and exposure-hours rasters from stage 1.
2. Filter the admin0 shapefile to the requested `location_id`.
3. Intersect the admin polygon with the storm's RR > 0 area; if no
   intersection, skip.
4. Compute:
   - `max_wind_speed` — the maximum intensity pixel inside the admin polygon
     (vectorized via `rasterize` → admin-id label raster → per-id max).
   - `person_storm_hours` — Σ over admin pixels of `population × exposure_hours`
     using the gridded population raster at 100 m.
   - `population_exposed` — Σ of population over the same exposed pixels.
   - `total_population` — admin0 FHS denominator (full population, not just
     exposed).
5. Write one row to a per-(storm, loc) parquet.

#### Script

`04_admin_level_exposure_b_main.py`

#### Runtime scope

One invocation processes **one** 9-tuple:
`(source_id, variant_label, experiment_id, batch_year, basin, draw, storm_id,
location_id)`. Each task writes a single per-storm-per-location parquet
(one row). No internal parallelism — fan-out is launcher-level.

#### CLI

```
python 04_admin_level_exposure_b_main.py \
  --source_id <str> \
  --variant_label <str> \
  --experiment_id <str> \
  --batch_year <YYYY-YYYY> \
  --basin <EP|NA|NI|SI|AU|SP|WP> \
  --draw <int>                  # CLIMADA inner draw (0..99)
  --storm_id <int>              # zero-padded internally to 4 digits
  --location_id <int>           # admin0 loc_id
  --num_cores <int>             # CLI-only; not used internally
```

#### Inputs

| Path | Purpose |
|---|---|
| `<stage1>/<source>/<variant>/<exp>/<batch_year>/<basin>/intensity/intensity_*.zarr/storm_NNNN/` | Stage 1 storm intensity raster (one zarr group per storm) |
| `<stage1>/<source>/<variant>/<exp>/<batch_year>/<basin>/exposure_hours/exposure_hours_*.zarr/storm_NNNN/` | Stage 1 storm exposure-hours raster |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0.parquet` | Admin0 polygons for non-NA basins |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0_normalized.parquet` | Admin0 polygons for the NA basin (antimeridian-normalized, with special regions globbed into their admin 0 parents). Generated by `fix_missing_location_shapes.py` — see **Pre-stage — globbing special regions into the NA-basin admin 0 parquet**. Set via `SHP_ROOT_NORMALIZED`. |
| `/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2023_all_years.parquet` | FHS population totals (admin denominator) (set via `POP_TOTALS_PATH`) |
| `/mnt/team/rapidresponse/pub/population-model/results/2026_05_16/world_cylindrical_100/<year>q1.tif` | Gridded population, 100 m, ESRI:54034 (set via `GRIDED_POP_PATH`) |

#### Outputs

`SAVE_ROOT = /mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b/`

One parquet per task (one row per file):
```
<SAVE_ROOT>/<source>/<variant>/<exp>/<batch_year>/<basin>/tc_risk_draw_<draw>/storm_exposure/<year>/
  storm_<storm_id>_loc_<location_id>_<basin>_<source>_<variant>_<exp>_<start>01_<end>12<draw_text>.parquet
```

Columns:
| Column | Notes |
|---|---|
| `draw` | int, the CLIMADA inner draw |
| `storm_id` | str, the storm's zarr group key digit |
| `year` | int, derived from the storm's `start_date` |
| `location_id` | admin0 `loc_id` |
| `person_storm_hours` | Σ over exposed pixels of `pop × exposure_hours` |
| `total_population` | FHS admin total for the year (sentinel `1.0` if missing) |
| `total_population_exposed` | Σ pop over exposed pixels |
| `max_wind_speed` | max intensity (m/s) within the admin polygon |
| `special_region_flag` | `True` if `total_population` came from the FHS missing-row sentinel |

The save function also chmods file + parent to `0o775`.

#### Pipeline flow (`process_single_storm`)

1. Open the storm's intensity zarr (the storm's group inside the per-draw
   intensity zarr store). Read `start_date` → `year`.
2. Open the storm's exposure-hours zarr.
3. Load the basin-appropriate admin0 shapefile and filter to
   `loc_id == location_id` (≤ 1 row remains). If the column is missing,
   return early.
4. For NA basin: normalize datasets to -180..180.
5. Build a minimal lat/lon-extent template raster via
   `generate_storm_template_raster` (cheap, not basin-sized).
6. `ensure_min_grid` to guarantee ≥ 3×3 pixels on each axis.
7. Rasterize intensity + exposure to the template via `to_raster` +
   `resample_to`. Close the source datasets.
8. `clean_raster` on each (0 → NaN, crop to affected bbox).
9. `intersect_shapefile_with_rasters` → at most 1 admin polygon since the
   shapefile is pre-filtered. If empty, return.
10. Vectorized per-admin max wind: `rasterize` the admin polygons to a
    label raster matching the intensity grid, then `intensity[label == i]`
    + finite check + max per id.
11. `reproject_shapefile_to_equal_area` — normalize longitudes if needed,
    reproject the admin shape(s) to `ESRI:54034`, polygonize. No CEA-bbox
    clipping happens here; bbox trimming is done per-piece inside the
    admin loop below.
12. For each admin (per-admin try/except so one bad polygon doesn't kill
    the task):
    a. `split_antimeridian_geom` → one or two (CEA, WGS84) pairs.
    b. For each piece:
       - For non-NA basins: shift the WGS84 piece to 0-360 to match the
         stage 1 raster's lon convention.
       - `exposure.clip(piece_wgs84).mask(piece_wgs84, all_touched=True)`.
       - Guard against zero-dim clipped arrays (CEA-boundary edge case).
       - Compute the piece's CEA-bounded bbox; load 100-m population
         bounded to that bbox.
       - `mask(piece_cea, all_touched=True)` on population.
       - `resample_to` exposure onto the population grid; mask exposure
         on `piece_cea` too.
       - `valid_mask = (pop > 0) & isfinite(pop) & (exposure > 0) & isfinite(exposure)`.
       - Accumulate `person_storm_hours_total += Σ(pop × exposure)` and
         `population_exposed_total += Σ(pop)`.
    c. Look up FHS admin total via `get_population_total(pop_df, year, admin_id)`.
       Returns `(1.0, True)` sentinel for missing rows.
    d. Append a single row to `storm_records`.
13. Save the records to parquet via `save_storm_exposure` (chmods on save).

#### Resume semantics

**There is no per-task resume check inside the script.** Each invocation
processes its (storm, loc) regardless of disk state. Resume is delegated
entirely to the launcher: the launcher's filesystem scan should skip
tasks whose parquet already exists.

(Earlier versions had a `check_if_storm_is_complete` helper but it had a
filename mismatch — the writer included `_loc_<location_id>_` and the
check did not — so even uncommented it always returned False. Removed in
the refactor to avoid the misleading guard.)

##### Launcher-level resume (per-task, bulk-walk)

The launcher (`04_admin_level_exposure_b_launcher.py`) submits one Jobmon
task per `(source_id, variant_label, experiment_id, batch_year, basin,
draw, storm_id, location_id)` 8-tuple — drawn from
`stage4a_metadata_admin0/resource_estimation_all_storms.parquet`
(produced by Part A's resource-assignment downstream task). At 100K+
tasks per launcher invocation, per-row filesystem checks would be too
slow.

Instead, `gather_completed_tasks(meta_df)` does a **bulk walk** of the
output tree:

1. For each unique `(source/variant/exp/batch_year/basin/draw)` combo
   referenced in `meta_df`, find the `tc_risk_draw_<draw>/storm_exposure/`
   directory under `SAVE_ROOT`.
2. `rglob("storm_*_loc_*.parquet")` across all year subdirectories
   (year isn't in the launcher's input — it's derived from `start_date`
   inside the worker, so we glob with `**` instead of constructing the
   exact path).
3. For each file ≥ 1 KB, parse `storm_id` and `location_id` from the
   filename via a regex (`^storm_(\d+)_loc_(\d+)_`).
4. Insert into a set of completed 8-tuples.

After the walk, `meta_df` is filtered in one pass via set-membership:
```python
_meta_keys = list(zip(meta_df["source_id"], ..., meta_df["location_id"]))
_completed_mask = pd.Series([k in _completed_keys for k in _meta_keys], ...)
meta_df = meta_df[~_completed_mask].copy()
```

The launcher prints a `Completion scan: N / M tasks already done; K to
submit.` line so re-run behavior is visible.

**Why the bulk walk is needed:** because the worker doesn't have a
per-task resume check anymore (the broken `check_if_storm_is_complete`
was removed), and because the output path includes a `<year>/` segment
the launcher doesn't have. A naive per-row glob would do 100K+ stat
walks; the bulk approach does one walk per unique draw and uses set
lookups for the per-task check.

#### Performance characteristics

- **No full-raster reprojection** — per-piece, the CEA bbox used for
  bounding the population read is obtained by taking the WGS84 piece,
  intersecting it with the raster's WGS84 `.bounds + 0.2°` buffer box,
  then `.to_crs("ESRI:54034").bounds`. Only 4 corners go through pyproj
  instead of every pixel. The full per-pixel reprojection that the
  pre-refactor `reproject_raster_to_equal_area` did is gone.
- **Minimal storm template raster** — `generate_storm_template_raster`
  allocates only what the storm's lat/lon extent needs plus a small
  buffer, instead of a basin-wide grid.
- **Vectorized per-admin max wind** — one `rasterio.features.rasterize`
  pass with admin-id labels, then `intensity[label == i].max()` per admin.
- **Bounded population reads** — `load_in_gridded_population(bounds=...)`
  reads only the CEA window of the 100 m population tiff per piece.
- **Per-admin `try/except`** isolates polygon-level failures so the task
  saves whatever admins succeeded rather than crashing the whole task.
- **No explicit `gc.collect()`** — refcount-based reclamation handles the
  linear `del` chain.
- **Save chmods its own file** (`0o775` on file + parent) — no post-task
  tree walk.

#### Failure modes worth knowing

- **Storm zarr group missing** — `get_exposure_storm_from_draw` raises
  `FileNotFoundError`; the script returns early without writing a parquet.
  The launcher should detect the missing output and either retry or
  blacklist that (storm, loc).
- **Admin polygon doesn't intersect stage 1 RR mask** — `intersected_shapes`
  comes back empty; task returns with no parquet written. Same launcher
  follow-up applies.
- **CEA-boundary edge cases** — small admins flush against ±180° can
  produce zero-dimension clipped rasters. Guard at the per-piece level
  skips just that piece (the admin's other pieces, if any, still
  contribute).
- **Per-admin failures** — wrapped in `try/except`; the failing admin is
  logged with full traceback and skipped. Since the shapefile is
  pre-filtered to one `location_id`, in practice this means the whole
  task's parquet is empty rather than partial.
- **FHS missing-row sentinel** — `get_population_total` returns
  `(1.0, True)` when no row matches the `(location_id, year)` query.
  Downstream consumers should filter on `special_region_flag` before
  aggregating, since the population-weighted ratio for those rows is
  effectively the raw exposed person-hours, not a true population
  fraction.

#### Post-processing downstream task

Once every per-(storm × loc) task completes, a single downstream Jobmon
task walks the 4B output tree and writes one consolidated parquet
containing every row that Part B produced. At ~35M expected outputs
(100 draws × all combos × all (storm, loc) pairs), an `rglob` walk would
be slow and produces no inventory of *missing* files. Instead, paths are
enumerated up-front from 4A's compiled metadata (which already carries
the `year` segment that the worker derives at runtime).

##### Script

`04_admin_level_exposure_b_post_processing.py`

##### Inputs

| Path | Purpose |
|---|---|
| `stage4a_metadata_admin0/compiled_admin_level_metadata.parquet` | One row per (storm × admin × draw) — the inventory of every expected 4B output, including `year` |
| Every `<SAVE_ROOT>/<source>/<variant>/<exp>/<batch_year>/<basin>/tc_risk_draw_<draw>/storm_exposure/<year>/storm_<storm_id>_loc_<location_id>_*.parquet` | The per-task outputs to consolidate |

##### Output

```
/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b/_consolidated/
  storm_exposure_all.parquet
```

One row per (storm × loc × draw × combo) — the union of every Part B
parquet that exists on disk. Missing 4B parquets are silently skipped
(safe against partially-completed runs).

##### Constants

| Constant | Value | Notes |
|---|---|---|
| `COMPILED_4A_PARQUET` | `stage4a_metadata_admin0/compiled_admin_level_metadata.parquet` | Path source of truth (35M-row inventory) |
| `SAVE_ROOT` | `stage4b/` | Same root the worker writes to |
| `CONSOLIDATED_DIR` | `stage4b/_consolidated/` | Leading-underscore subdir so it sorts above combo subdirs |
| `ID_COLS` | 9 path-build columns including `year` | Projected from the 4A compile; `year` is the only one 4B's launcher loses |
| `CHUNK_SIZE` | 5000 | Rows per parallel-read chunk (~7K chunks at 35M rows) |
| `NUM_CORES` | 40 | Parallel reader pool size |

##### Pipeline flow

1. **`enumerate_meta()`** — `pd.read_parquet(COMPILED_4A_PARQUET, columns=ID_COLS)`
   loads the 35M-row inventory in projected form (no other columns). One
   row per expected 4B output.
2. **`compile_exposure(meta_df)`** — slice the inventory into chunks of
   `CHUNK_SIZE` rows and dispatch via `run_parallel`. Each worker:
   - Builds the 4B output path on the fly with `_build_4b_path(row)`
     (mirrors `save_storm_exposure`'s filename construction exactly).
   - `pd.read_parquet(path)`; `FileNotFoundError` is skipped silently,
     other exceptions are logged and skipped.
   - Concats its chunk and returns one DataFrame.
3. Final concat across all chunk DataFrames → write to
   `CONSOLIDATED_PARQUET`.

Paths are **never** materialized as a list of `Path` objects — chunks
are DataFrame slices and path strings are built lazily inside each
worker. Peak memory is dominated by the final concat (~10-15 GB for
35M rows × ~9 columns).

##### Wiring into the 4B launcher

The 4B launcher creates a single `CLIMADA_stage4b_post_processing` task
at the end of task creation, with `upstream_tasks=tasks` and a
parameterless command template (just `python <SCRIPT>`).

- If 4B is fully done on disk and the launcher's bulk-walk resume scan
  filters `tasks` to empty, the post-processing task still runs (no
  upstreams = fires immediately).
- The launcher adds `[post_processing_task]` to the workflow
  unconditionally, after the conditional `add_tasks(tasks)` for the
  per-(storm × loc) tasks.

Resource budget for the post-processing task (tunable in the launcher):
- `cores = 20` (Jobmon-allocated; the script's internal `NUM_CORES = 40`
  controls the parallel-read pool size, which can over-subscribe)
- `memory = 100G`
- `runtime = 240m`
- `max_attempts = 2`

The runtime budget is the rough one — reading 35M small parquets is
I/O-bound and the actual wall time depends heavily on filer load. Bump
cores first if it's too slow; bump memory only if the final concat OOMs.

##### Failure modes worth knowing

- **Missing 4B parquets** — `_read_chunk` catches `FileNotFoundError`
  and skips. The consolidated parquet ends up with a row for every 4B
  parquet that *exists*; missing rows correspond to either (a) tasks
  that failed permanently, (b) tasks the worker returned early from
  (e.g., admin polygon doesn't intersect the storm's RR mask), or (c)
  4A inventory entries that aren't real 4B tasks. The launcher's own
  bulk-walk completion scan is the canonical source of "what succeeded".
- **4A compile checkpoint missing** — `enumerate_meta()` fails fast
  with `FileNotFoundError`. The 4A bridge's `compile_metadata` writes
  the checkpoint as a side effect of its run, so this only fires if 4A
  was wiped or the bridge was skipped.
- **Final concat OOM** — at ~35M rows the concat peaks around 10-15 GB.
  Default budget (100G) has comfortable headroom, but if rows-per-parquet
  ever grows (e.g., the worker starts writing multiple admin rows per
  task), the final concat is the first thing that'll break.

### Part A — Per-(storm, admin) metadata-gathering pass

#### Purpose

Quantify the **work envelope** of each (storm × admin) cell before the
heavy 4B compute runs. For every storm in every year of a draw, this stage
records pixel counts, percent affected, area, and bounding box per admin
polygon that the storm intersects. The output drives 4B's launcher-level
resource-budgeting (which (storm, location_id) tasks to submit, with what
memory/runtime).

For each storm × admin:
1. Identify which admin polygons intersect the storm's RR > 0 area.
2. For each admin polygon, split at the antimeridian and per-piece:
   - Load 1km gridded population bounded to the piece's reprojected bbox.
   - Resample exposure-hours to the population grid.
   - Build a validity mask (`pop > 0 & exposure > 0`).
   - Count total pixels, affected pixels, and accumulate affected
     population.
3. Save one row of metadata per (storm × admin) with pixel counts, area,
   bbox, and processing time.

After all storms in a year are done, concat into a yearly metadata parquet.
After all years in the draw are done, concat into a draw-level parquet.

#### Script

`04_admin_level_exposure_a_main.py`

#### Runtime scope

One invocation processes one 7-tuple
`(source_id, variant_label, experiment_id, batch_year, basin, draw_batch,
admin_level)` and parallelizes over draws inside it. Each parallel worker
handles one full draw (all years × all storms × all admins for that draw).

Parallelism is launcher-controlled via `num_cores`, with one draw per
worker.

#### CLI

```
python 04_admin_level_exposure_a_main.py \
  --source_id <str> \
  --variant_label <str> \
  --experiment_id <str> \
  --batch_year <YYYY-YYYY> \
  --basin <EP|NA|NI|SI|AU|SP|WP> \
  --draw_batch <start-end>         # e.g. "0-9"
  --admin_level <0|1|2> \
  --num_cores <int>                # default 5; parallelism is over draws
```

#### Inputs

| Path | Purpose |
|---|---|
| `<stage1>/<source>/<variant>/<exp>/<batch_year>/<basin>/intensity/intensity_*.zarr/storm_NNNN/` | Stage 1 storm intensity raster |
| `<stage1>/<source>/<variant>/<exp>/<batch_year>/<basin>/exposure_hours/exposure_hours_*.zarr/storm_NNNN/` | Stage 1 storm exposure-hours raster |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin<N>.parquet` | Admin polygons at the requested admin_level for non-NA basins |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0_normalized.parquet` | NA-basin admin **0** polygons (antimeridian-normalized, with special regions globbed into their admin 0 parents). Generated by `fix_missing_location_shapes.py` — see **Pre-stage — globbing special regions into the NA-basin admin 0 parquet**. Set via `SHP_PATH_NORMALIZED_A0`. |
| `/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29/lbd_standard_admin_<N>.shp` | NA-basin admin polygons for `admin_level > 0` (legacy LBD .shp); set via `SHP_ROOT_NORMALIZED_HIGHER` |
| `/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2023_all_years.parquet` | FHS population totals (kept for parity with 4B; admin filter here doesn't apply age/sex) (set via `POP_TOTALS_PATH`) |
| `/mnt/team/rapidresponse/pub/population-model/results/2026_05_16/world_cylindrical_1000/<year>q1.tif` | Gridded population, **1 km** (coarser than 4B's 100 m — this is a metadata pass) (set via `GRIDED_POP_PATH`) |

#### Outputs

`SAVE_ROOT = /mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4a_metadata_admin{admin_level}/`

Three nested parquets per draw:

```
<SAVE_ROOT>/<source>/<variant>/<exp>/<batch_year>/<basin>/tc_risk_draw_<draw>/
  storm_metadata/<year>/storm_<storm_id>_<basin>_<source>_..._{draw_text}.parquet
  yearly_metadata/<year>/yearly_metadata_<year>_<basin>_<source>_..._{draw_text}.parquet
  admin_level_metadata/admin_level_metadata_<basin>_<source>_..._{draw_text}.parquet
```

Per-row columns (saved in the per-storm parquet; aggregated unchanged
upward):

| Column | Notes |
|---|---|
| `source_id`, `variant_label`, `experiment_id`, `batch_year`, `basin` | Task identifiers |
| `storm_id` | str, the storm's zarr group key digits |
| `year` | int |
| `location_id` | admin polygon's `loc_id` |
| `projection` | the CRS of the population raster used (ESRI:54034) |
| `resolution` | string label `"estimated 100m from 1km"` |
| `total_population_pixels_100m` | `total_1km × 100` (the 1km count scaled to a 100m-equivalent figure) |
| `affected_population_pixels_100m` | `affected_1km × 100` |
| `percent_affected_100m` | `affected_1km / total_1km × 100` |
| `population_affected` | Σ population over `pop > 0 & exposure > 0` pixels |
| `xmin`, `xmax`, `ymin`, `ymax` | combined CEA bbox of all valid pieces |
| `area_100m2` | sum of piece areas in m² |
| `processing_time_seconds` | per-admin wall time (stringified, two decimals) |

Each save function chmods file + parent to `0o775`.

#### Pipeline flow

Per-invocation (`main`):
1. Parse `draw_batch` → list of draws.
2. Build the basin template raster once.
3. Fan out via `run_parallel(num_cores=num_cores)` — one worker per draw.

Per-draw, per-worker (`process_single_draw`):
1. **Resume short-circuit:** if `check_if_draw_is_complete` returns True
   (admin-level metadata parquet exists + valid), return.
2. Resolve stage 1 intensity + exposure zarr stores.
3. Load the basin-appropriate admin shapefile (parquet for non-NA, .shp
   for NA) at the requested `admin_level`.
4. Load the FHS pop totals (kept for column parity even though this stage
   doesn't filter age/sex like 4B does).
5. Enumerate storm metadata for the draw via `iter_storms_metadata` and
   bucket storms by year via `map_storms_to_years`.
6. For each year (skip if `check_if_year_is_complete`):
   a. For each storm (skip if `check_if_storm_is_complete`):
      - Open the storm zarr.
      - For NA basin: normalize datasets to -180..180.
      - `ensure_min_grid` on both.
      - Rasterize to the basin template via `to_raster` + `resample_to`.
      - `clean_raster` (0 → NaN, crop to affected bbox).
      - `intersect_shapefile_with_rasters` → admin polygons touching the
        storm.
      - Vectorized per-admin **max wind** via per-polygon clip + nanmax.
      - `reproject_shapefile_to_equal_area` → admin shapes reprojected
        to `ESRI:54034` and polygonized (no bbox trim in the helper;
        per-piece bbox clipping happens below).
      - **Per-admin loop** (try/except per admin, see below): split at
        antimeridian, per-piece clip exposure, intersect with raster
        bounds, load 1km population, mask + resample exposure to pop
        grid, count valid pixels and accumulate. Append one record per
        admin to `storm_records`.
      - Save per-storm metadata parquet.
   b. After all storms in the year: `save_yearly_exposure` concats the
      per-storm parquets into one yearly parquet.
7. After all years: `save_draw_dataframe` concats yearly parquets into
   one draw-level admin metadata parquet.

Per-storm and per-admin loops are wrapped in `try/except`: one bad storm
or polygon logs traceback + `continue`, the rest of the work proceeds.

#### Resume semantics

Three layered completion checks, each backed by a per-parquet file (size
> 0 + parquet header validates + ≥ 1 row):
- `check_if_storm_is_complete(storm_id, year, draw, ...)` — per-storm
  parquet under `storm_metadata/<year>/`.
- `check_if_year_is_complete(year, draw, ...)` — yearly aggregation
  under `yearly_metadata/<year>/`.
- `check_if_draw_is_complete(draw, ...)` — draw-level aggregation
  under `admin_level_metadata/`.

The hierarchical check matches the hierarchical writes: a complete
draw → skip the worker entirely; otherwise skip already-done years;
within a year, skip already-done storms; recompute the rest.

##### Launcher-level resume (per-draw-batch)

The launcher (`04_admin_level_exposure_a_launcher.py`) submits one Jobmon
task per `(source/variant/exp/batch_year/basin/draw_batch)` 6-tuple.
Completion is derived from the filesystem, not Jobmon:

- `_stage4a_draw_metadata_path(...)` builds the path of a single draw's
  admin-level metadata parquet — matches what `save_draw_dataframe`
  writes in the main script.
- `task_is_complete(row)` returns True iff every draw in the
  `draw_batch` has its parquet on disk and ≥ 1 KB. Mirrors
  `check_if_draw_is_complete` from the main script.

Tasks where any draw is missing or undersized get submitted; the main
script's three-layer in-script resume (storm / year / draw) then
short-circuits already-done work within each draw worker.

**Production overrides:** the launcher uses single-draw batches
(`DRAW_BATCHES = [f"{i}-{i}" for i in range(100)]`) and forces
`req_runtime_min = 3.0` / `memory_req = "2G"` regardless of the
`stage4_resource_requirements.parquet` estimates. Stage 4A is a
metadata-estimation pass — heavy resource budgeting belongs to 4B.


#### Performance characteristics

- **`itertuples(index=False)`** on both admin loops (max-wind pass +
  main calculations).
- **Per-storm + per-admin `try/except`** so one bad storm/polygon doesn't
  kill the draw worker (which would also skip `save_yearly_exposure` and
  `save_draw_dataframe`).
- **No explicit `gc.collect()`** — refcount reclamation on `del`.
- **Lazy equal-area bbox** — the per-piece CEA bbox is computed inline
  by intersecting the WGS84 piece with the raster's WGS84 `.bounds`
  (plus a 0.2° buffer) and reprojecting just the resulting box to
  `ESRI:54034`. Pyproj only ever transforms 4 corners per piece, not
  every pixel of the intensity + exposure rasters. See "Known precision
  shift" below.
- **Tight chmod** on every saved parquet (file + parent → `0o775`).

#### Failure modes worth knowing

- **Storm zarr group missing** — the per-storm try/except logs and
  continues to the next storm. The year still aggregates whatever storms
  did succeed.
- **Per-admin failure** (weird geometry, reprojection error) — logged
  with traceback; the storm's parquet still saves with whatever admins
  succeeded.
- **CEA-boundary edge cases** — `split_antimeridian_geom` rejects
  wrap-around CEA pieces (width > 10,000 km is the threshold) to avoid
  spurious global-spanning geometries. A clamping warning is printed if
  any piece sits right at `CEA_MAX_X` / `CEA_MIN_X`.

##### Known precision shift (post-refactor)

The pre-refactor code computed the CEA bbox by **warping every pixel**
of the intensity and exposure rasters via `rasterra.to_crs()`, then
reading the resulting raster's `.bounds`. The refactored code does a
**corner transform** inline: take the WGS84 piece, intersect with
`box(*admin_exposure.bounds) ± 0.2°`, `.to_crs("ESRI:54034")`, and read
`.bounds`. Pyproj only transforms 4 corners per piece.

Both are valid; they differ at the sub-pixel level because the old
approach implicitly **quantized to rasterra's CEA pixel grid (~1 km)**,
while the new approach is geometrically exact. Observed delta in the
saved metadata, on a representative storm × admin:

| Field | Old (quantized) | New (exact) | Δ |
|---|---|---|---|
| `xmin` | -8,938,955 | -8,938,955 | 0 |
| `xmax` | -8,639,955 | -8,637,955 | +2,000 m (≈ 2 CEA px) |
| `ymin` | 3,454,727 | 3,454,727 | 0 |
| `ymax` | 3,602,727 | 3,602,727 | 0 |
| `total_population_pixels_100m` | 4,425,200 | 4,454,800 | +0.67% |
| `affected_population_pixels_100m` | 1,534,800 | 1,536,800 | +0.13% |

The wider eastern bbox propagates through the per-piece exposure clip →
`intersection_cea_bounds` → bounded `load_in_gridded_population` window
→ `pop_piece_masked.bounds`. Both totals shift slightly higher (the
exact bbox captures sub-pixel area the quantized bbox truncated).

We picked the exact behavior because (a) this is a metadata-estimation
stage feeding 4B's resource budget — sub-1% delta is below the
meaningful resolution for that decision, and (b) the per-pixel warp cost
wasn't justified by the precision difference. If 4A output ever feeds a
non-budgeting consumer that's bit-sensitive, restore the per-pixel
`rasterra.to_crs` path or quantize the corner-transform bbox to a 1 km
CEA grid.

#### Resource-assignment downstream task

Once every per-(combo × draw) metadata task completes, a single downstream
Jobmon task compiles 4A's `admin_level_metadata` parquets and writes the
per-(storm, location_id) resource-estimation parquet that Part B's
launcher consumes. This is the bridge between the 4A metadata pass and
the 4B compute pass.

##### Script

`04_admin_level_exposure_a_resource_assignment.py`

##### Inputs

| Path | Purpose |
|---|---|
| Every `<SAVE_ROOT>/<source>/<variant>/<exp>/<batch_year>/<basin>/tc_risk_draw_<draw>/admin_level_metadata/*.parquet` written by `save_draw_dataframe` | One row per (storm × admin) cell — drives size-class binning |
| `/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv` | The (source/variant/exp/batch_year/basin) combos to walk |

##### Outputs

```
/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4a_metadata_admin0/
  compiled_admin_level_metadata.parquet     # checkpoint: full concat
  resource_estimation_all_storms.parquet    # 4B's input (one row per (storm × loc))
```

The compiled parquet is a checkpoint so `assign_resources` can be re-run
on its own without redoing the I/O (e.g. retuning `RESOURCE_MAP` without
re-walking 100K+ parquets).

##### Constants

| Constant | Value | Notes |
|---|---|---|
| `DRAWS` | `range(100)` | All inner draws |
| `BASELINE_BATCH_YEAR` | `"1965-1969"` | Excluded — historical baseline isn't run in stage 4 |
| `PIXEL_SIZE_M` / `PIXEL_AREA_M2` | 100 / 10,000 | Used to convert `area_100m2` → n_pixels |
| `CHUNK_SIZE` | 1500 | Per-chunk parallel-read size |
| `NUM_CORES` | 40 | Parallel reader pool size |
| `LOC_ID_OVERRIDE_VERY_LARGE` | 22 | Force this `location_id` into the `"very_large"` bucket regardless of pixel count |
| `PIXEL_BINS` | `[0, 1e6, 2e7, 2e8, 5e8, 1.5e9, ∞]` | n_pixels bin edges |
| `SIZE_CLASSES` | `[very_small, small, medium, large, very_large, extreme]` | 6 size classes |
| `RUNTIME_BINS` / `RUNTIME_LABELS` | `[0, 1, 2, 4, 6, ∞]` / `[1, 2, 4, 6, 8]` | Defensive runtime ceiling — current `RESOURCE_MAP` emits `{1, 2, 4, 6}` only |

##### `RESOURCE_MAP` (v14)

| Size class | Memory | Runtime |
|---|---|---|
| `very_small` | 3 GB | 1 min |
| `small` | 4 GB | 1 min |
| `medium` | 5 GB | 2 min |
| `large` | 10 GB | 2 min |
| `very_large` | 29 GB | 4 min |
| `extreme` | 70 GB | 6 min |

v14 changes from prior versions:
- 3-min runtimes collapsed → 2 min across the map.
- `extreme` bucket runtime 4 → 6 min.
- `loc_id == 22` forced into `very_large` regardless of pixel count
  (empirically heavier than its pixel count suggests).

##### Pipeline flow

1. **`enumerate_paths()`** — cross-join `level_4_task_assignments.csv`
   with draws 0..99, drop the baseline batch, and emit a list of
   `(parquet_path, draw)` pairs.
2. **`compile_metadata(pairs)`** — `run_parallel` reads the parquets in
   chunks of `CHUNK_SIZE`, stamps each row with its draw, and concats
   into one DataFrame. Missing files are logged and skipped (safe against
   a partially-completed 4A). Saves the checkpoint parquet.
3. **`assign_resources(df)`** — derives `n_pixels` from `area_100m2`,
   bins into `SIZE_CLASSES` via `pd.cut(bins=PIXEL_BINS)`, applies
   `RESOURCE_MAP`, applies the `LOC_ID_OVERRIDE_VERY_LARGE` override,
   runs the defensive runtime re-bin, drops to one row per
   `(storm × loc)` via `drop_duplicates(subset=task_cols)`, and writes
   the launcher-ready parquet.

##### Wiring into the 4A launcher

The 4A launcher creates a single `CLIMADA_stage4a_resource_assignment`
task at the end of task creation, with `upstream_tasks=tasks` (every
per-(combo × draw) task) and a parameterless command template (just
`python <SCRIPT>`).

- If 4A is fully done on disk and the launcher's resume scan filters
  `tasks` to empty, the resource-assignment task still runs (no
  upstreams = fires immediately).
- The launcher adds `[resource_assignment_task]` to the workflow
  unconditionally, after the conditional `add_tasks(tasks)` for the
  per-(combo × draw) tasks.

Resource budget for the assignment task (tunable in the launcher):
- `cores = 20` (Jobmon-allocated; the script's internal `NUM_CORES = 40`
  controls the parallel-read pool size, which can over-subscribe)
- `memory = 20G`
- `runtime = 60m`
- `max_attempts = 2`

##### Failure modes worth knowing

- **Missing 4A parquets** — `_read_parquet_chunk` catches
  `FileNotFoundError` and logs; the affected (storm × admin) rows simply
  don't contribute. If many are missing, 4B will under-estimate task
  counts and `resource_estimation_all_storms.parquet` won't contain rows
  for them.
- **`area_100m2 > PIXEL_BINS[-1]` (1.5e9)** — `pd.cut` bins into
  `"extreme"`. If a real (storm × loc) ever exceeds 70 GB / 6 min,
  re-tune `PIXEL_BINS` + `RESOURCE_MAP`.
- **Schema drift in 4A's metadata parquet** — the bridge depends on
  `area_100m2`, `location_id`, and the 6 task-identifier columns.
  Adding columns is safe; renaming any of those breaks the bin step
  with a KeyError on the lookup.
