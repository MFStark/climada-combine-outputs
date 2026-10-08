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
team-mount parquets reflect the new release. Note that neither pre-stage
rebuilds `global_WGS84_admin0.parquet`, the admin 0 file used for
non-NA basins — see **Which shapefile each stage reads**.

Note: the `/ibtracs/` folder mirrors most — but not all — of the main
scripts. The two Stage 4 downstream-task scripts
(`04_admin_level_exposure_a_resource_assignment.py` and
`04_admin_level_exposure_b_post_processing.py`) are **main-only**; the
ibtracs side has no matching files for them. See **Stage 4 — ibtracs
differences** below.

## Output versions

Every pipeline output root carries a version suffix so a rerun never reads or
resumes from an older run's files. Bump the suffix in all 14 pipeline
scripts (stages 1–4, launchers, the two Stage 3 metadata generators, the 4A
resource-assignment step and the 4B post-processing step) to start a new
version.

| Version | Produced | Code | Status |
|---|---|---|---|
| **v3** | not yet run | current scripts (includes the 2026-10-08 review fixes: Stage 1 grid fix and draw markers, failure handling in stages 1–4, antimeridian and population-window fixes, post-processing manifest) | **current**; every section below describes v3 paths |
| v2 | Mar–Jun 2026 | earlier scripts (see each stage's **Provenance** subsection) | production data on disk; affected by the Stage 1 grid-stretch bug, so superseded by v3 |

v3 locations under `/mnt/team/rapidresponse/pub/tropical-storms/climada/output/`:

| Output | v2 | v3 |
|---|---|---|
| Stage 1 zarrs / draw markers | `stage1_v2/`, `stage1_v2_log/` | `stage1_v3/`, `stage1_v3_log/` |
| Stage 2 GeoTIFFs | `stage2_v2/` | `stage2_v3/` |
| Stage 3 task metadata | `storm_draw_admin{0,1}_count.parquet` | `storm_draw_admin{0,1}_count_v3.parquet` |
| Stage 3 parquets | `stage3_v2/`, `stage3_v2_admin1/` | `stage3_v3/`, `stage3_v3_admin1/` |
| 4A metadata, inventory, resource parquet, grouped task lists | `stage4a_metadata_admin{0,1}/` | `stage4a_metadata_v3_admin{0,1}/` |
| 4B outputs, sentinels, consolidated parquet | `stage4b_v2/`, `stage4b_v2_admin1/` | `stage4b_v3/`, `stage4b_v3_admin1/` |

The 4A metadata root and the Stage 3 metadata parquets were unversioned in
v2. They are versioned in v3 because both are resumed from disk: the 4A
worker skips any draw whose metadata parquet exists, and the Stage 3
launcher only regenerates its metadata parquet when it is missing.

Inputs are not versioned (CMIP6 tracks, shapefiles, population, RR tables,
`storm_draw_table.csv`). Two non-pipeline scripts still point at v2:
`03_admin_level_paf_launcher_modifed_memory.py` (old variant launcher) and
`script/log_scripts/stage3_check_main.py`.

---

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

`script/population.py` (one level up from this directory, not in
`script/climada/`)

### Inputs

| Source | Notes |
|---|---|
| GBD 2023 `get_population` via `db_queries` | `release_id = 16`, `age_group_id = 22` (all ages), `sex_id = 3` (both sexes), `year_id = 1970..2100` requested (GBD only returns actuals, i.e. through 2024), location IDs from the FHS hierarchy (`location_set_id = 39`) at `level <= 4` — countries **and** admin 1 subnationals, which the admin 1 runs of stages 3 + 4 need |
| FHS forecast NetCDF | `/mnt/share/forecasting/data/32/future/population/future_population_s130v41/population_agg.nc` — sliced to `sex_id = 3`, `age_group_id = 22`, draws mean-reduced |

### Outputs

```
/mnt/team/rapidresponse/pub/tropical-storms/
  fhs_population_2023.parquet              # GBD actuals only (1970..2024, one row per loc/year)
  fhs_population_2023_future.parquet       # FHS forecast only (2025..2100, draws-mean)
  fhs_population_2023_all_years.parquet    # concat of the two — POP_TOTALS_PATH target
```

The third file is the one stage 3 + 4 actually read. The other two are
intermediate checkpoints kept for diagnostic / re-aggregation use.

Verified contents of the current files (2026-06-16 build):

| File | Rows | Years | Locations |
|---|---|---|---|
| `fhs_population_2023.parquet` | 28,215 | 1970–2024 | 513 |
| `fhs_population_2023_future.parquet` | 38,988 | 2025–2100 | 513 |
| `fhs_population_2023_all_years.parquet` | 67,203 | 1970–2100 | 513 |

No duplicate `(location_id, year_id, age_group_id, sex_id)` keys, and all
513 locations are present in every year across the 2024 → 2025 hand-off.

### How to run

`db_queries` is an IHME shared library that requires the GBD conda
environment. Activate it first, then run the script:

```shell
source /ihme/code/central_comp/miniconda/bin/activate gbd_env
python ../population.py   # run from script/climada/
```

### When to re-run

Re-run whenever any of these change:
- GBD release version (e.g., 2023 → 2024) — bump `gbd_2023_release_id` in the script.
- FHS forecast version — bump the `future_population_s130v41` path.
- Year range or age/sex filter — edit the `get_population(...)` call.

**Hand-off years are hardcoded.** The FHS-side drop is the literal
`df[~df['year_id'].isin([2023, 2024])]`, not derived from the last year
GBD returned. On a release whose actuals end in a different year, update
that list (or replace it with `df[df.year_id > all_population.year_id.max()]`),
otherwise the output silently gains a duplicate year or loses one. After
any rebuild, re-check that no location × year key is duplicated or missing.

After generating a new parquet, point the pipeline at it by updating
`POP_TOTALS_PATH` in all 6 stage 3 + 4 main scripts (3 climada + 3 ibtracs).

---

## Pre-stage — generating the 0-360 admin shapefiles

This script converts the LBD admin shapefiles from the EPSG:4326
[-180, 180] convention to **[0, 360]** longitudes, matching CLIMADA's
basin centroid convention. In the current pipeline **only its admin 1
output is consumed**; the admin 0 file that stages 3 + 4 read for
non-NA basins comes from elsewhere (see **Which shapefile each stage
reads** below).

### Script

`generate_0_360_shapefiles.py`

### Inputs

| Path | Purpose |
|---|---|
| `/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29/lbd_standard_admin_<N>.shp` | LBD admin polygons in [-180, 180]; N ∈ `ADMIN_LEVELS` = `[0, 1]` |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/fhs_hierarchy_39_32.csv` | FHS location hierarchy (`FHS_LOC_PATH`). Admin 1 only: rows are filtered to `loc_id`s at FHS `level == 4` (`filter_to_fhs_admin1`), so every admin 1 polygon has FHS population and no special-region merge is needed |

### Outputs

```
/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/
  global_WGS84_admin0_0_360.parquet     # 256 rows, special regions NOT merged — not read by any stage
  global_WGS84_admin1_0_360.parquet     # 280 rows, FHS-filtered — read by stages 3 + 4 when admin_level == 1
```

The script writes to a `_0_360` suffix (`OUTPUT_FILENAME_TEMPLATE`) so the
canonical `global_WGS84_admin{N}.parquet` files aren't clobbered. Admin 2
is no longer generated (it was dropped from `ADMIN_LEVELS`); the
`global_WGS84_admin2.parquet` on the team mount (2026-04-17) is an older
artifact that no stage reads.

### Which shapefile each stage reads

Verified against the code and the files on disk (2026-10-08):

| File | Date | Rows | Lon | Special regions | Produced by | Read by |
|---|---|---|---|---|---|---|
| `global_WGS84.parquet` | 2026-02-27 | 256 | 0–360 | not merged | likely an earlier run of this script (byte-identical size and same rows as the `_0_360` output); pre-glob snapshot | Stage 1, non-NA basins |
| `global_WGS84_admin0.parquet` | 2026-03-10 | 206 | 0–360 | merged | **unknown — no script in this repo** | Stages 3 / 4A / 4B, admin 0, non-NA basins |
| `global_WGS84_admin0_0_360.parquet` | 2026-06-16 | 256 | 0–360 | not merged | this script | nothing |
| `global_WGS84_admin0_normalized.parquet` | 2026-05-28 | 209 | −180–180 | merged | `fix_missing_location_shapes.py` | Stages 3 / 4A / 4B, admin 0, NA basin |
| `global_WGS84_admin1_0_360.parquet` | 2026-06-16 | 280 | 0–360 | n/a (FHS-filtered) | this script | Stages 3 / 4A / 4B, admin 1, **all** basins (NA is normalized to −180..180 at load time) |

`global_WGS84_admin0.parquet` has the same 47 special regions merged in as
the NA file, but it also drops three locations the NA file keeps
standalone: Vatican City (353), Paracel Islands (60931), and Spratly
Islands (93924). Its provenance isn't recorded; if it needs rebuilding
for a new LBD release, the closest reproducible path is this script's
`_0_360` admin 0 output followed by the `SPECIAL_TO_PARENT` merge from
`fix_missing_location_shapes.py`.

### Reprojection logic

`reproject_geometry` per polygon:
- Entirely in [0, 180] → unchanged.
- Entirely in [-180, 0] → shifted +360.
- Crosses the prime meridian normally → split at lon = 0, west half shifted +360.
- Crosses the antimeridian (touches ±180) → clipped at lon = 0, west half
  shifted, re-joined with a 1e-4 buffer to close the numerical seam.

Known edge case: `shift_left_piece` / `shift_normal` only handle
`Polygon` and `MultiPolygon` and return any other geometry type
unchanged. If `part.intersection(LEFT_BOX)` ever yields a
`GeometryCollection` (possible when a polygon runs along x = 0), that
piece would be left at negative longitudes. It does not occur with the
`2024_07_29` release — no row in any 0–360 parquet has `minx < 0` — but
re-check after an LBD bump.

### How to run

Sequential, no parallelism — sized for a one-off run after an LBD update.

```shell
conda run -n climada_env python generate_0_360_shapefiles.py
```

By default the script writes to `..._0_360.parquet` so existing canonical
files aren't clobbered. Do **not** retarget `OUTPUT_FILENAME_TEMPLATE` to
`"global_WGS84_admin{admin_level}.parquet"` to replace the admin 0 file
stages 3 + 4 read: this script does not merge special regions, so the
result would reintroduce the 47 child `loc_id`s that the current file has
merged into their parents.

### When to re-run

Re-run whenever:
- The LBD shapefile root bumps (currently `2024_07_29`) — edit `INPUT_ROOT`.
- The FHS hierarchy changes (affects the admin 1 filter) — edit `FHS_LOC_PATH`.
- A new admin level is needed downstream — add to `ADMIN_LEVELS`.

Stages 3 + 4 read `global_WGS84_admin1_0_360.parquet` by name, so a
regenerated admin 1 file is picked up with no downstream edits. A
regenerated admin 0 `_0_360` file is not picked up by anything (see the
table above).

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

This file is used for the **NA basin at admin 0 only**. Non-NA basins
read `global_WGS84_admin0.parquet`, which has the same 47 children merged
but has *dropped* Vatican City, Paracel Islands and Spratly Islands
rather than keeping them standalone (206 rows vs 209 here). For admin 1,
every basin reads `global_WGS84_admin1_0_360.parquet` instead.

Two of the standalones, Liechtenstein (360) and Antarctica (60921), have
no row in the FHS population file, so if they were ever hit they would
get the `(1.0, special_region_flag=True)` population sentinel in stages
3 + 4. Neither is exposed to tropical storms. Paracel and Spratly also
have no FHS population.

### Script

`script/fix_missing_location_shapes.py` (one level up from this
directory, not in `script/climada/`)

### Inputs

| Path | Purpose |
|---|---|
| `/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29/lbd_standard_admin_0.shp` | LBD admin 0 polygons (already antimeridian-normalized in EPSG:4326); set via `SHP_ROOT_NORMALIZED` + `INPUT_PATH` |

### Outputs

```
/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/
  global_WGS84_admin0_normalized.parquet
```

Row count: input rows minus the 47 globbed children (256 → 209 for
`2024_07_29`; verified on disk). Lon range is −180..180 (Russia's `maxx`
is 180.0 plus floating-point noise, which is harmless). The script will
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
conda run -n climada_env python ../fix_missing_location_shapes.py   # run from script/climada/
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

`storm_draw_admin{0,1}_count_v3.parquet` is the metadata file consumed by the
Stage 3 launcher (`03_admin_level_paf_launcher.py`) to enumerate tasks and
size their resources. One file per admin level; the launcher reads the one
matching its `ADMIN_LEVEL`.

Each file has one row per
`(storm_draw, source_id, variant_label, experiment_id, batch_year, basin)`
for **one representative storm draw per (source_id, variant_label)**, i.e.
the 8 lowest-numbered storm draws (one per model × variant pair), crossed with
every combination except the 1965–1969 baseline: **3,612 rows**. The launcher
then spreads each row's metrics to all 100 storm draws by joining on the
5-tuple `(source_id, variant_label, experiment_id, batch_year, basin)`. The
metrics only drive resource sizing. The **set of 5-tuples** in this file is
what defines the Stage 3 task list.

The Stage 3 launcher checks for this file at startup and **automatically
runs the generation script** if it is missing. It does **not** regenerate a
file that exists, so after re-running Stages 1–2 (or pointing them at new
versioned output roots) delete the stale file or re-run the script by hand,
and point its `STAGE1_ROOT` / `PAF_ROOT` / `OUTPUT_PATH` constants at the new
version (both generators currently read `stage1_v3` / `stage2_v3` and write
`storm_draw_admin{0,1}_count_v3.parquet`; see **Output versions**).

### Scripts

| Admin level | Script | Output | Shapefile used | Metric columns |
|---|---|---|---|---|
| 0 | `generate_storm_draw_admin0_count.py` | `storm_draw_admin0_count_v3.parquet` | `global_WGS84_admin0.parquet` | `num_admin0_first_year`, `estimated_admin0_total` (counts `ADM0_CODE`) |
| 1 | `generate_storm_draw_admin1_count.py` | `storm_draw_admin1_count_v3.parquet` | `global_WGS84_admin1_0_360.parquet` | `num_admin1_first_year`, `estimated_admin1_total` (counts `ADM1_CODE`) |

The two scripts are otherwise identical.

### Inputs

| Path | Purpose |
|---|---|
| `/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv` | All (model, variant, scenario, time_period, basin) combos to process |
| `/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv` | Maps storm draws to (source_id, variant_label) and sample columns |
| `stage1_v3/<source_id>/<variant_label>/<experiment_id>/<batch_year>/<basin>/intensity/intensity_*_<start>01_<end>12.zarr` | **Draw 0** Stage 1 intensity store only; `storm_XXXX/` group count = `n_storms_in_batch` |
| `stage2_v3/<storm_draw>/…/<first year>/<basin>/raw_paf/draw_mean_raw_paf_*indirect_cvd_draw*.tif` | Stage 2 **CVD** PAF raster for the **first year** of the batch; used to count intersecting admin polygons |
| admin shapefile (see table above) | Polygons counted against the PAF > 0 area |

### Output columns

| Column | Description |
|---|---|
| `storm_draw` | e.g. `storm_draw_0002` |
| `source_id`, `variant_label`, `experiment_id`, `batch_year`, `basin` | Task key |
| `direct_rr_draw`, `indirect_cvd_draw`, `indirect_resp_draw` | Sample names from the storm draw table |
| `n_storms_in_batch` | Count of `storm_XXXX/` groups in the draw-0 Stage 1 intensity store |
| `estimated_storms_per_year` | `n_storms_in_batch / num_years_in_batch` |
| `year` | First year of the batch (string) |
| `num_admin{N}_first_year` | Unique admin regions intersecting the first-year CVD PAF raster |
| `num_years_in_batch` | Number of years in the batch period |
| `estimated_admin{N}_total` | `num_admin{N}_first_year × num_years_in_batch` (the launcher's resource-bin key) |

### Output path

```
/mnt/team/rapidresponse/pub/tropical-storms/climada/output/storm_draw_admin0_count_v3.parquet
/mnt/team/rapidresponse/pub/tropical-storms/climada/output/storm_draw_admin1_count_v3.parquet
```


### How to run

```shell
python generate_storm_draw_admin0_count.py
python generate_storm_draw_admin1_count.py
```

Uses 64 cores via `rra_tools.parallel.run_parallel`. Expect ~30–60 minutes
depending on filesystem load. File permissions are set to `0o775` on write.

### Known limitations

- **NA counts are undercounted.** Both scripts intersect 0–360 shapefiles
  with every basin's PAF raster, but NA rasters are in −180..180, so only the
  sliver where the two overlap is counted (admin 0 median 5 vs dozens of real
  NA locations; admin 1 median 2). This affects resource binning only: NA
  tasks land in the ≤ 100 bin. Task enumeration is unaffected.
- **Draw 0 must have a Stage 1 intensity store** for every combination;
  otherwise `count_storms_in_batch` raises `FileNotFoundError` and the whole
  run fails (`run_parallel` has no per-row error handling).
- `sd.iloc[:8]` assumes exactly 8 (source_id, variant_label) pairs in the
  storm draw table; adding a pair requires changing that constant.

### When to re-run

Re-run when any of these change:
- Stage 1 zarr data is regenerated (changes `n_storms_in_batch`).
- Stage 2 PAF rasters are regenerated (changes `num_admin{N}_first_year`).
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

### Provenance of the production outputs (read first)

The current `01_climada_intensity_main.py` **consolidates** the earlier
Stage 1 variants so that the NA basin runs correctly in a single pass. It has
not yet produced the production data. What is on disk today was written by
earlier code:

| Output on disk | Written | Code |
|---|---|---|
| `stage1_v2/` (all basins) + `stage1_v2_log/draw_completion_markers/` | ~2026-03-20 | `old/01_climada_intensity_main.py` |
| `stage1_v2/` NA basin, storms whose footprint reached lon ≥ 359.9 | ~2026-03-26 | `01_climada_intensity_main_na_rerun.py` (git `a08249b6` only) |
| `stage1_v2_storm_country_log/` | ~2026-03-20 | `old/01_climada_intensity_main.py` (`save_country_storms`) |

Known issues in that production data, all fixed in the current code:

- **Grid stretch (~68% of stored storms).** The old crop used
  `dropna(how="all")`, which also removed interior empty rows/cols. `to_raster`
  then averaged the coordinate step, so values were stored, and land-clipped,
  at the wrong locations: up to ~330 km for the peak, and up to 62 m/s per pixel.
  Every stage that reads Stage 1 rasters inherits this. Before/after figure:
  `notebooks/2026_10_08_claude_fixes.ipynb`.
- **NA mixed longitude convention.** Only the re-run storms (~10%) are stored in
  −180..180; the rest are in 0–360. Downstream NA normalization
  (`((lon + 180) % 360) - 180`) is idempotent, so this is handled. A few
  exposure arrays reach lon ≥ 359.9 in 0–360 (the exposure footprint at 17 m/s
  is slightly larger than the intensity footprint at 17.5 m/s that the re-run
  selected on); land just east of 0° was dropped for those.
- **Failed storms were never retried.** The draw marker was written even when
  storms raised, and the old code had no per-storm sentinel, so a failed storm
  cannot be told apart from "no landfall" on disk.
- **No landfall parquets.** The `metadata/.../landfall_locations/` parquets
  below come from the current code only. The old code wrote one
  `country_storms_*.csv` per draw (parquet bytes despite the extension), and
  each storm overwrote the previous one, so only the last landfalling storm
  per draw survives. Nothing downstream reads it.

The rerun writes to `stage1_v3/` + `stage1_v3_log/` (see **Output versions**), so it starts with no markers.

### Script
`01_climada_intensity_main.py` (worker) and `01_climada_intensity_launcher.py`

### Runtime scope

Unique draw-runs per launcher submission =
`N_combinations × 100 draws`, where `N_combinations` is the row count of
`(source_id, variant_label, experiment_id, batch_year, basin)` in
`level_4_task_assignments.csv` (currently **3,668** → **366,800** draws).

### Launcher

- Draws are batched `DRAWS_PER_BATCH = 5` per task (`"0-4"`, …, `"95-99"`),
  giving 73,360 tasks; each task runs its 5 draws in parallel (`num_cores = 5`).
- Fixed `memory = 35G`. Runtime is binned from the storm count of the
  combination's **draw 0** NetCDF (`n_trk`): bin values 40/50/80/120/190/250 min
  for ≤15/≤30/≤70/≤120/≤180/>180 storms, plus a 10-minute buffer (50–260 min).
- Submission order finishes one (source, variant) before the next, with
  `MRI-ESM2-0` (many variants) pushed to the end (`DEPRIORITIZED_SOURCES`).
- `workflow.run()` exceptions are printed, not re-raised (launcher exits 0).

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

Exit code is **1** if any storm in any draw of the batch failed (see Resume
semantics), 0 otherwise.

### Inputs

| Path | Purpose |
|---|---|
| `/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/<source_id>/<variant_label>/<experiment_id>/<batch_year>/<basin>/tracks_*.nc` | Synthetic TC tracks (one NetCDF per draw; multi-storm along `n_trk`). Winds in knots (`max_sustained_wind_unit = kn`), hourly steps (`time_step = 1`), `time` = seconds since the 1st of `tc_month`. |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84.parquet` | Land polygons (0-360 lon) for all basins except NA. This is the **pre-glob** snapshot (256 rows, special regions not merged). Stages 3 + 4 read `global_WGS84_admin0.parquet` instead (206 rows, merged), so the two are expected to differ; see **Which shapefile each stage reads** in the pre-stage section. |
| `/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29/lbd_standard_admin_0.shp` | Land polygons (−180..180 lon, normalized) for the NA basin |

Both gdfs share columns `loc_id` and `ADM0_NAME` used downstream.

Input data notes (verified 2026-10-08):
- Some tracks have **trailing** steps with `lon`/`lat`/`environmental_pressure`
  present but `max_sustained_wind`/`central_pressure` NaN. They are kept by the
  any-finite trim; CLIMADA sets NaN windfields to 0, so they add no wind.
- The NetCDF `valid_storm` flag is not read; it agreed with the script's own
  all-core-vars-finite scan on every sampled draw.
- Tracks do leave the basin grid (basin bounds + 5°), in up to ~9% of WP storms,
  but only after weakening below 34 kn (≈17.5 m/s), so no above-threshold wind is lost.

### Outputs

All outputs live under
`/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1_v3/`.

| Output | Path (relative to `stage1_v3/`) |
|---|---|
| Intensity zarr | `<source>/<variant>/<exp>/<batch_year>/<basin>/intensity/intensity_<basin>_<source>_<exp>_<variant>_<start>01_<end>12<draw_text>.zarr/storm_NNNN/` |
| Exposure zarr  | `<source>/<variant>/<exp>/<batch_year>/<basin>/exposure_hours/exposure_hours_<basin>_<source>_<exp>_<variant>_<start>01_<end>12<draw_text>.zarr/storm_NNNN/` |
| Landfall parquet | `metadata/<source>/<variant>/<exp>/<batch_year>/<basin>/landfall_locations/landfall_<basin>_<source>_<exp>_<variant>_<start>01_<end>12<draw_text>/storm_NNNN.parquet` |
| Draw completion marker | `../stage1_v3_log/draw_completion_markers/<source>/<variant>/<exp>/<batch_year>/<basin>/draw_NNNN.json` |

Conventions:
- `<draw_text>` is empty for draw 0; otherwise `_e{draw-1}` (CMIP ensemble member suffix).
- Zarr group key per storm is `storm_{storm_index:04d}` (4-digit zero-padded).
- Zarr v3 with blosc/zstd level 9 + bitshuffle, chunked `(lat=64, lon=64)`, fp32.
- Each storm array is on the **0.1° basin grid**, cropped to the bounding box of
  its footprint (interior empty rows/cols are kept as NaN). Coordinates carry
  float noise (e.g. `97.40000000000103`); round before joining on them.
- `exposure_hours` carries `units = "hours"`, which makes xarray decode it
  as `timedelta64`. Open with `decode_timedelta=False` to get the float hours.
- Landfall parquet columns: `source_id, variant_label, experiment_id, batch_year, basin, draw, storm_id, start_date, loc_id, ADM0_NAME, max_wind_m_s`.

### Constants

| Constant | Value | Notes |
|---|---|---|
| `RESOLUTION` | `0.1` degrees | Centroid grid spacing |
| Centroid `buffer_deg` | `5.0` | Added to each basin's lon/lat bounds |
| Land subset `buffer` | `0.25` degrees | Pad applied to storm bbox before spatial join |
| Wind threshold | `17.0` m/s | Used by landfall check, exposure, and per-loc max filter |

Threshold note: CLIMADA's `TropCyclone.from_tracks` zeroes `haz.intensity`
below its default `intensity_thres = 17.5` m/s, while exposure hours are counted
from the raw windfields at > 17 m/s. So the intensity footprint (and the
landfall check built on it) effectively starts at 17.5 m/s, and the exposure
footprint is slightly larger. The stated pipeline threshold remains > 17 m/s.

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
  Tracks crossing 180° are shifted to −180..180 before CLIMADA; CLIMADA
  re-centres both track and centroid longitudes per track, so this is safe.
- Intensity and exposure outputs are **normalized to −180..180 for NA only**
  after CLIMADA emits them (via `normalize_lon_to_180`). This matches the
  normalized NA shapefile used for land clipping.
- All other basins remain in 0-360 end-to-end.
- Production NA data is mixed (see Provenance).

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
   doesn't kill the draw; failing storm indices are collected.
8. After the loop, chmod the written zarr stores. Write the draw completion
   marker **only if no storm failed**; otherwise return the failed indices.

`main` gathers the failed indices from every draw and exits 1 if any draw is
incomplete.

Per-storm (`process_single_storm`):

1. **Resume check** — `check_existing_storm_in_zarr` returns True if the
   landfall parquet exists and (if non-empty) the zarrs also validate.
2. Slice the storm from the open NetCDF
   (`read_single_storm_from_dataset` — trims to first/last ANY-finite step).
3. Normalize to CLIMADA's track schema (`normalize_nc_storm_for_climada`).
   `radius_max_wind` is passed as 0, so CLIMADA estimates it from pressure.
4. `TropCyclone.from_tracks(... store_windfields=True)` — heaviest step.
5. Compute intensity grid (`generate_intensity_per_storm`) and crop it to the
   footprint bounding box (`crop_to_footprint`).
6. For NA: `normalize_lon_to_180(storm_intensity)`.
7. Spatial-index lookup for the storm's land polygons
   (`subset_land_for_storm` — uses `gdf.sindex.query` with `intersects`).
8. Storm-level landfall check (`check_storm_landfall` — fast union geom).
9. Clip intensity to land → save intensity zarr.
10. Per-polygon zonal max → landfall records (held in memory).
11. Compute wind-speed cube → yearly exposure hours
    (`compute_yearly_exposure_per_storm`, also cropped with `crop_to_footprint`).
12. Clip exposure to land → save exposure zarr.
13. Save the landfall parquet **last**, so the sentinel only exists once both
    zarrs are written.

`to_raster` raises `ValueError` if the lat/lon coordinates are not uniformly
spaced, instead of averaging a gap into a stretched resolution. Inside a storm
this is caught by the per-storm `try/except`, so the storm is recorded as
failed and the draw is left unmarked.

### Case handling

| Case | Condition | Intensity zarr | Exposure zarr | Landfall parquet |
|---|---|---|---|---|
| **0 Resume** | Sentinel parquet says "done" | (skipped) | (skipped) | (already on disk) |
| **1 NoIntensity** | CLIMADA produced zero wind at every centroid | — | — | empty |
| **2 No land in bbox** | Storm's buffered bbox doesn't intersect any polygon | — | — | empty |
| **3 No landfall** | No wind > 17 m/s OR no strong winds reach land | — | — | empty |
| **4 Landfall** | `check_storm_landfall` returns True; ≥1 polygon clears 17 m/s | written | written | rows |
| **4b Edge** | Storm-level passes but no single polygon clears 17 m/s | written | written | empty |
| **5 Error** | Any other exception in `process_single_storm` | (maybe partial) | (maybe partial) | — → draw left unmarked, task exits 1, Jobmon retries |

### Resume semantics

The **landfall parquet is the per-storm completion sentinel**:

- **Missing** → storm not yet processed (or failed); full pipeline runs.
- **Empty** → storm was processed and found to make no qualifying landfall;
  no zarrs are expected (or case 4b, where both zarrs were written before the
  sentinel). Storm short-circuits on subsequent runs.
- **Non-empty** → storm makes landfall; zarrs must also exist and pass
  structural validation (`check_existing_storm_in_zarr`). If a zarr is
  missing or has a `.partial` file, that storm's zarr group is deleted and
  the storm is reprocessed.

Draw-level completion is marked separately by a JSON file in
`stage1_v3_log/draw_completion_markers/...`, written only when **every**
storm in the draw succeeded. A draw is skipped at the top of
`process_single_draw` if its marker exists. Per-storm sentinels handle
partial-draw recovery; the draw marker handles "skip the whole draw".

A failed storm therefore leaves its draw unmarked and makes the task exit 1.
Jobmon retries the task in the same workflow run (up to its `max_attempts`;
the launcher does not set it, so Jobmon's default applies),
and the retry recomputes only storms without a sentinel. A storm that fails
deterministically will exhaust the retries and surface as a failed task in
the workflow, rather than being silently dropped.

#### Launcher-level resume (batch granularity)

The launcher (`01_climada_intensity_launcher.py`) submits **draw-batches**
(e.g., `"0-4"` = draws 0..4). It derives completion by scanning
`stage1_v3_log/draw_completion_markers/...` for per-draw markers and treats a
batch as done iff every draw in it has a marker.

If a batch crashes part-way (e.g., 3 of 5 draws marked), the launcher
**reruns the whole batch**. The cost of re-submitting a partially-done batch
is bounded: inside the rerun, already-marked draws return immediately and
the per-storm `check_existing_storm_in_zarr` short-circuits every
already-done storm, so the only real work is the failed or missing storms.

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

- **Non-uniform grid** — `to_raster` raises `ValueError` listing the
  offending lat/lon steps. With `crop_to_footprint` in place this should not
  happen; if it does, the storm fails loudly (no sentinel, draw unmarked).
- **CLIMADA grid-mismatch** (`ValueError` from `generate_intensity_per_storm`
  or `generate_speed_per_storm`) — propagates to the storm-level try/except,
  logs full traceback, no sentinel written, draw unmarked, task exits 1.
- **`.partial` files** — orphans from a killed Python process mid-write.
  `check_and_cleanup_zarr_store` deletes such zarr stores at the start of
  each draw (note: it returns early if the intensity store is absent, so it
  never inspects the exposure store in that case); `check_existing_storm_in_zarr`
  deletes individual storm groups if it finds a `.partial` inside.
- **NA shapefile read latency** — `lbd_standard_admin_0.shp` lives on
  `/snfs1` and is slower to read than the team-mount parquet. Stages 3
  and 4 already migrated to the antimeridian-normalized parquet at
  `global_WGS84_admin0_normalized.parquet`; Stage 1 still reads the
  `.shp` directly. Migrating Stage 1 to read the same parquet would
  shave a few seconds off every draw's startup.
- **Half-pixel raster registration** — `to_raster` places the affine origin
  at the first pixel *centre* (not its corner), so rasters are treated as
  0.05° east/south of their coordinates when clipped. Accepted: the
  convention is the same in every stage, and the final land clip happens
  downstream.

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

### Provenance of the production outputs

`stage2_v2/` was written ~2026-03-26. Re-running the current
`02_relative_risk_main.py` functions in memory on a production task
(NI, `storm_draw_0081`, CVD, 2027) reproduces the saved GeoTIFF exactly
(max diff 0). So the code matches the production outputs, but those outputs
are built on the grid-stretched Stage 1 rasters (see Stage 1 Provenance) and
need re-running after Stage 1. The current code refuses those stretched
inputs (see Pipeline flow).

Verified 2026-10-08 on a sample: 0 missing or undersized GeoTIFFs across
600 tasks; values in [0, 0.032] with no NaN or negative pixels.

### Script

`02_relative_risk_main.py` (worker) and `02_relative_risk_launcher.py`

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

`storm_draw_table.csv` has 100 storm draws. Each maps to one
`(source_id, variant_label)` (1–29 storm draws per pair) and names one RR
sample column per cause (`indirect_cvd_draw`, `indirect_resp_draw`). All storm
draws of the same (source, variant) read the **same** Stage 1 zarrs; they
differ only in the RR sample. Launcher scope: the task combinations minus the
1965–1969 baseline, × storm draws, × 2 causes = **90,874 tasks**,
**424,200 GeoTIFFs**.

### Launcher

- Resources: `num_cores = 10`, `memory = 4G`. Runtime comes from
  `stage2_resource_usage.parquet`, read from a **personal** path
  (`/mnt/share/homes/mfiking/downloads/climada_rs/`), binned to
  5/10/20/30/45/60/95-minute ceilings, then **×3** (applied on every
  launch, not only reruns).
- Combinations missing from the resource parquet get `fillna(30)`. The value is
  in **seconds** (the comment says 60 minutes), so they land in the 5-minute
  bucket (15 min after ×3).
- `PRIORITY_MODE = "all"`: every storm draw, with `PRIORITY_DRAWS` first.

### CLI

```
python 02_relative_risk_main.py \
  --storm_draw <storm_draw_NNNN>      # selects output path; RR sample comes from --sample_name
  --source_id <str> \
  --variant_label <str> \
  --experiment_id <str> \
  --batch_year <YYYY-YYYY> \
  --basin <EP|NA|NI|SI|AU|SP|WP> \
  --relative_risk <indirect_resp_draw|indirect_cvd_draw> \
  --sample_name <str>                 # column name in the RR table
  --num_cores <int>
```

Exit code is **1** if any storm failed in any inner draw (see Resume
semantics), 0 otherwise.

### Inputs

| Path | Purpose |
|---|---|
| `/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1_v3/<source>/<variant>/<experiment>/<batch_year>/<basin>/intensity/intensity_*.zarr/storm_NNNN/` | Stage 1 per-storm intensity rasters (one zarr group per storm, one zarr store per inner draw) |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/relative_risk_samples/rd_rr_samples.csv` | Respiratory-disease RR curve (`relative_risk=indirect_resp_draw`) |
| `/mnt/team/rapidresponse/pub/tropical-storms/data/relative_risk_samples/cvd_rr_samples.csv` | Cardiovascular-disease RR curve (`relative_risk=indirect_cvd_draw`) |

Both RR CSVs have 21 rows and columns `windspeed`, `type`, `mean`, `min`,
`max`, `sample_001` … `sample_100`; one sample column is selected via
`--sample_name`. They are byte-identical to the copies in the repo's `data/`
folder used by the March production run.

**Wind-speed units (left as-is by decision, 2026-10-08).** The `windspeed`
column runs 12.8611 → 64.3055 in steps of 2.5722, which is exactly
25 → 125 kn expressed in m/s. `build_rr_interpolator` treats the column as
knots and multiplies by 0.514444 again, so the curve is applied over
6.6–33.1 m/s. On one test task, removing that second conversion lowered the
year's PAF by 33% (CVD) and 21% (respiratory). Revisit if the RR source
confirms the column is in m/s.

### Outputs

`SAVE_ROOT = /mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage2_v3/`

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
- Grid: the basin template (0.1°, basin bounds + 5°; NA in −180..180). The
  template stops at the buffered bound, while Stage 1's centroid grid extends
  one more 0.1° row/col; that edge row/col is dropped (negligible).

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
downstream stages may renormalize. (No negative pixels were found in the
sampled production outputs.)

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
   `process_single_draw` via `run_parallel`. Each worker returns
   `(yearly_paf, failures)`. Sum `yearly_paf[year]` into `cumulative_paf[year]`
   and collect the `(draw, year, storm_id)` of every failure.
7. Divide by `n_draws = 100` to get the draw-mean.
8. Save one GeoTIFF per invalid year **that has no failed storm**; collect
   the saved paths.
9. Single chmod sweep over `(file, file.parent)` for every saved path.
10. If any storm failed, print the failures and exit 1.

Per inner draw, per worker (`process_single_draw`):

1. Build the per-draw `RRInterpolator` from the shipped `rr_samples_df` +
   `sample_name` (`build_rr_interpolator`). Reused for every storm.
2. Locate the stage 1 intensity zarr (`get_draw_zarr_path`). If missing or
   empty → return zero rasters for all invalid years (no failures).
3. Read all storm metadata once (`iter_storms_metadata`) to map storms to
   the years they affect (`map_storms_to_years`). A storm is assigned to every
   year from its `start_date` year to its `end_date` year.
4. For each year:
   a. If the year has no storms → fill with zeros and continue.
   b. For each storm in that year (try/except per storm; on failure log
      traceback, record `(year, storm_id)` and `continue`):
      - Open the storm zarr in a `with` block (handle released on any exit).
      - `ensure_min_grid` to guarantee ≥ 3×3 pixels for rasterization.
      - For NA basin: `normalize_dataset` (0-360 → −180..180; idempotent, so
        the mixed production NA conventions are handled).
      - Compute RR via `generate_relative_risk` + the pre-built interpolator.
      - Mask non-positive RR to NaN (`where(rr_da > 0)`).
      - Convert to `rt.RasterArray` with `to_raster(..., expected_res=template
        res)`, which raises unless the coords step is exactly 0.1° in both
        axes. Pre-fix Stage 1 rasters are evenly spaced at the **wrong** step
        (e.g. 0.115°), so they fail here instead of being smeared across the
        template.
      - Resample (nearest) to the basin template.
      - Mask `= rr_values > 0` (NumPy `>` excludes NaN). If any: accumulate
        `(rr − 1)/rr × IMPACT_DAYS_FRACTION` into `sum_raw_paf`.

### Resume semantics

The completion sentinel is **per-year GeoTIFF existence + validity**:

- **Missing** → year not yet computed (or not saved because a storm failed);
  included in `invalid_years`.
- **Size < 1 KB** → corrupt; file is unlinked, year goes to `invalid_years`.
- **Fails `rt.load_raster`** → unreadable; file is unlinked, year goes to
  `invalid_years`.
- **All years valid** → script returns immediately at the top of `main` via
  `check_if_draw_complete`.

A year in which any storm failed (in any of the 100 inner draws) is **not
saved**, so it stays invalid. The task exits 1, Jobmon retries it, and the
retry recomputes only the unsaved years. Years without failures are saved on
the first attempt.

If a single year is invalid, it is recomputed by re-running all 100 inner
draws *for that year*. There is no per-(inner draw) sentinel — the inner-draw
accumulator runs in memory and is only persisted as a mean.

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

- **Stretched (pre-fix) Stage 1 inputs** — every affected storm raises in
  `to_raster`, its years are not saved, and the task exits 1. Re-running
  Stage 2 against the current production `stage1_v2` fails by design; re-run
  Stage 1 first.
- **Storm-level failure** — logged with traceback; the affected year is not
  saved, the task exits 1, and Jobmon retries. A deterministic failure
  exhausts the retries and shows as a failed task.
- **Unreadable storm metadata** — `iter_storms_metadata` has no try/except,
  so a bad storm group raises out of the worker and fails the whole task.
- **Stage 1 zarr missing or empty** — `process_single_draw` returns zero
  rasters for that inner draw rather than failing (a draw with no landfalling
  storms has no store). This cannot distinguish "no landfall" from "missing
  Stage 1 output", so Stage 1 must be complete (all draw markers) first.
- **Storms spanning two calendar years** contribute their full 20-day PAF to
  both years. None were found in the sampled data (every exposure array is
  single-year).
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
of **population-weighted PAF per admin location** (admin 0 or admin 1). For
each admin polygon that intersects the basin's affected area:

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

### Provenance of the production outputs

`stage3_v2/` (admin 0) was written ~2026-06-03. Re-running the current
`process_single_year` in memory on production task-years reproduces the saved
parquets exactly, apart from the antimeridian fix below (see **Verification**).
The outputs are built on the grid-stretched Stage 1 rasters (see Stage 1
Provenance) and need re-running after Stages 1–2.

### Script

`03_admin_level_paf_main.py` (worker) and `03_admin_level_paf_launcher.py`

### Runtime scope

Same 8-tuple per invocation as stage 2:
`(storm_draw, source_id, variant_label, experiment_id, batch_year, basin,
relative_risk, sample_name)`, plus `--admin_level`. Inside:

- One year in `batch_year` per parallel worker via `run_parallel`, batched by
  `num_cores` (the launcher passes 1, so years run sequentially).
- Each worker computes population-weighted PAF for every admin polygon that
  intersects the basin's RR > 0 area for that year.

### Admin level

`ADMIN_LEVEL` in the launcher (currently **0**) selects the metadata parquet,
output root and resource table, and is passed to the worker as
`--admin_level`. Both levels are run for the full pipeline: launch once with
`ADMIN_LEVEL = 0` and once with `ADMIN_LEVEL = 1`.

| | Admin 0 | Admin 1 |
|---|---|---|
| Output root | `stage3_v3/` | `stage3_v3_admin1/` |
| Metadata parquet | `storm_draw_admin0_count_v3.parquet` | `storm_draw_admin1_count_v3.parquet` |
| Shapefile, non-NA basins | `global_WGS84_admin0.parquet` (0–360) | `global_WGS84_admin1_0_360.parquet` (0–360) |
| Shapefile, NA basin | `global_WGS84_admin0_normalized.parquet` (−180..180) | `global_WGS84_admin1_0_360.parquet`, normalized to −180..180 at load |

### CLI

```
python 03_admin_level_paf_main.py \
  --storm_draw <storm_draw_NNNN> \
  --source_id <str> \
  --variant_label <str> \
  --experiment_id <str> \
  --batch_year <YYYY-YYYY> \
  --basin <EP|NA|NI|SI|AU|SP|WP> \
  --relative_risk <indirect_resp_draw|indirect_cvd_draw> \
  --sample_name <str> \
  --num_cores <int> \
  --admin_level <0|1>        # default 0
```

Exit code is **1** if any admin failed in any year (see Resume semantics).

### Inputs

| Path | Purpose |
|---|---|
| `stage2_v3/<storm_draw>/.../<year>/<basin>/raw_paf/draw_mean_raw_paf_*.tif` | Stage 2 per-year PAF rasters (one per year in batch_year); 0–360 lon for non-NA basins, −180..180 for NA |
| admin shapefile (see **Admin level**) | Admin polygons. The admin 0 and admin 1 files for non-NA basins are in **0–360** lon (not −180..180). The NA admin 0 file is antimeridian-normalized with special regions globbed into their admin 0 parents (generated by `fix_missing_location_shapes.py`). |
| `/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2023_all_years.parquet` | FHS population totals by `location_id × year_id × age_group_id × sex_id` (set via `POP_TOTALS_PATH`) |
| `/mnt/team/rapidresponse/pub/population-model/results/2026_05_16/world_cylindrical_100/<year>q1.tif` | Gridded population, 100 m, ESRI:54034 (cylindrical equal area) (set via `GRIDED_POP_PATH`) |

The FHS query filters to `age_group_id = 22` (all ages) and `sex_id = 3`
(both sexes). Missing admin × year returns a sentinel population of `1.0`
with `special_region_flag = True` so the row is preserved with a clearly
flagged denominator.

Two admin 0 source polygons are invalid in the LBD shapefile itself
(`loc_id` 161 Bangladesh, 163 India). They still process (both appear in
every sampled NI output), but a GEOS topology error on them would now fail the
year loudly rather than drop the row.

### Outputs

`SAVE_ROOT` = `stage3_v3/` (admin 0) or `stage3_v3_admin1/` (admin 1), under
`/mnt/team/rapidresponse/pub/tropical-storms/climada/output/`.

One parquet per year:
```
<SAVE_ROOT>/<storm_draw>/<source>/<variant>/<exp>/<batch_year>/<basin>/<year>/paf_df/
  paf_<storm_draw>_<rr>_<sample>_<basin>_<source>_<exp>_<variant>_<start>01_<end>12_<year>.parquet
```

Per-row columns:
| Column | Notes |
|---|---|
| `storm_draw` | str, e.g. `storm_draw_0042` |
| `location_id` | admin `loc_id` from the shapefile |
| `year` | int, the four-digit year |
| `total_population` | FHS admin total for that year (sentinel `1.0` if missing) |
| `population_exposed` | Σ population within RR > 0 pixels (NOT the denominator) |
| `population_weighted_paf` | Σ(pop × paf) / `total_population` |
| `relative_risk` | the CLI `--relative_risk` value |
| `special_region_flag` | `True` if `total_population` came from the missing-row sentinel |

File mode `0o775`; parent dir `0o775` (set inside `save_batch_paf_dataframe`,
so each worker is self-contained for permissions).

`population_exposed` can exceed `total_population` slightly (≈12% of sampled
rows; at most +3%, median +0.04%). The 100 m gridded population sums a little
above the FHS totals, mostly for small islands and fully exposed countries.
This is expected, not a bug.

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
   `run_parallel(num_cores=num_cores)`. Each call returns the `location_id`s
   that failed (empty list on success).
3. Each worker writes its parquet + chmods it; no post-parallel sweep needed.
4. If any year returned failures, print them and exit 1.

Per-year, per-worker (`process_single_year`):
1. **Resume check** — `check_if_year_complete` returns True iff the parquet
   exists, is > 0 bytes, opens via `pyarrow.parquet.ParquetFile`, and has
   ≥ 1 row. If True, return.
2. Load the basin-appropriate admin shapefile + FHS population totals.
3. Load the stage 2 PAF raster + clean it (0 → NaN, crop to affected area
   via `subset_affected_area`).
4. `intersect_shapefile_with_raster` — find admin polygons whose footprint
   intersects RR > 0 pixels in the cleaned raster.
5. `reproject_shapefile_to_equal_area` — if any shape has `maxx > 180`,
   move the shapes to −180..180 with `normalize_longitudes` (see
   **Antimeridian handling**); reproject to `ESRI:54034`; polygonize.
6. For each admin polygon (try/except per admin; a failure records the
   `location_id` and continues):
   a. `split_antimeridian_geom` — produce one or two (CEA, WGS84) pairs
      depending on whether the polygon wraps.
   b. For each piece:
      - For non-NA basins: shift the WGS84 piece to 0-360
        (`normalize_geom_to_0_360`) to match the stage 2 PAF raster.
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
7. If any admin failed, **do not save**; return the failed ids.
8. Otherwise concat records into a sorted DataFrame and save to parquet
   (chmod inside the save function).

### Antimeridian handling

Non-NA admin shapefiles are in 0–360. `normalize_longitudes` moves them to
−180..180 before the equal-area reprojection:

- Geometries entirely west of 180 are unchanged; geometries entirely east of
  180 are translated by −360.
- Geometries with a ring that **straddles** 180 are cut at 180 in 0–360 space
  and only the eastern part is translated. In admin 0 this applies to **Russia
  (62, Chukotka) and Antarctica (60921)**; admin 1 has none. (Before
  2026-10-08 every vertex was shifted independently, which turned those rings
  into invalid shapes spanning 360°, which were then used as the population
  mask.) Areas are preserved exactly (relative difference ~1e-14).
- Multi-part countries whose parts sit on either side of 180 (e.g. Fiji) were
  already handled correctly and are unchanged.

`split_antimeridian_geom` then groups the parts west/east by centroid. For
non-NA basins each WGS84 piece is shifted back to 0–360 to clip the PAF
raster, while the population grid uses −180..180-based CEA coordinates. GDAL
wraps longitudes during `resample_to`, so pieces east of 180° (e.g. Mexico and
Central America in EP; Tonga, Samoa) get correct, non-zero PAF.

**Population window (`to_pm180`).** The 100 m population window for each
piece is the CEA bbox of the piece ∩ (PAF bounds + 0.2°). For non-NA basins
that intersection is in 0–360, and a piece starting **exactly at 180°** (the
eastern half of an island cut at the antimeridian, e.g. part of Fiji, or
Russia's Chukotka after the split above) has vertices at 180 and >180. pyproj
maps lon 180 to +20,037 km but lon 181 to about −19,900 km, so the bbox came
out **40,075 km wide** (the whole globe at that latitude) instead of a few
hundred km. Results were still correct (the population is masked to the piece),
but each such piece read a globe-wide strip of 100 m population. Since
2026-10-08, `to_pm180` translates a 0–360 geometry lying east of 180 by −360
before the CEA reprojection. Pieces starting just east of 180 (e.g. 180.01)
were never affected, because all their vertices wrap the same way. The same
fix is in 4A and 4B.

### Resume semantics

Per-year parquet is the sentinel. `check_if_year_complete` requires:
- file exists,
- size > 0 bytes,
- opens cleanly via `pyarrow.parquet.ParquetFile`,
- has ≥ 1 data row.

Any failure → that year goes into the worker queue. Already-valid years are
skipped at the top of `process_single_year`. There is no per-admin sentinel
— if a year is rerun, every admin polygon recomputes.

A year in which any admin raised is **not saved**, so it stays incomplete; the
task exits 1 and Jobmon retries it (`max_attempts = 5`, memory ×1.25 and
runtime ×1.75 per retry). A deterministic failure exhausts the retries and
shows as a failed task, rather than producing a parquet that silently lacks
that location. Errors outside the admin loop (e.g. a missing Stage 2 GeoTIFF)
propagate out of the worker and also fail the task.

A year with **no intersecting admin** returns without writing a parquet, so
the launcher would treat that task as incomplete on every launch. This did not
occur in production (0 missing year-parquets in a 1,500-task sample), because
a 100-draw mean almost always has some affected land.

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

The filesystem check keys on the full 8-tuple. (The earlier Jobmon-based
completion merged without `sample_name`, so one finished sample marked every
other sample of the same 7-tuple as done.)

Task list: the 5-tuples in the metadata parquet × all storm draws of the
matching (source_id, variant_label) × 2 causes.

Submission gating (`PRIORITY_MODE`, currently **`"all"`** for the full
rerun):
- `"non_priority"` — submit everything except the hardcoded `PRIORITY_DRAWS`
  list of 8 storm_draws.
- `"priority"` — submit only `PRIORITY_DRAWS` (smoke runs).
- `"all"` — submit every storm_draw; tasks are ordered round-robin across
  models by storm-draw rank.

Resources: 1 core. Memory/runtime come from a two-bin table keyed on
`estimated_admin{N}_total` from the metadata parquet (`pd.cut` bins
`(0, 100]` → `"heavy"`, `(100, ∞)` → `"light"`):

| Level | `(0, 100]` | `> 100` | `0` or missing |
|---|---|---|---|
| 0 | 50 G / 50 min | 50 G / 35 min | 10 G / 10 min |
| 1 | 4 G / 8 min | 10 G / 5 min | 10 G / 10 min |

Notes: the admin 1 block sets 8 G / 8 min defaults, but the
`DEFAULT_MAX_RUN_TIME_MIN` / `DEFAULT_MEMORY_GB = 10` lines after the
`ADMIN_LEVEL` block override them, so the effective default is 10 G / 10 min.
The `"heavy"` label is on the smaller bin. Retries scale memory ×1.25 and
runtime ×1.75 (`default_resource_scales`), up to `max_attempts = 5`.

### Verification (2026-10-08)

In-memory runs of the current `process_single_year` on production task-years
(`storm_draw_0081`, MRI-ESM2-0 r4i1p1f1 ssp585 2025–2029, CVD, 2027),
compared location by location with the saved parquets:

| Basin | Result |
|---|---|
| NI | 43/43 locations; identical except Ethiopia (179) at 6e-6 relative (run-to-run drift, see below) |
| WP | 24/24; identical except China (6) at the 6th significant figure (same drift). **Russia unchanged** by the antimeridian fix |
| SP | identical, 12/12 locations (incl. Fiji) |
| NI, admin 163 forced to fail | year not saved; `[163]` returned → task exits 1 |
| SP 2027, with vs without `to_pm180` | identical values; peak memory **46.0 → 20.5 GB**, 125 → 69 s |
| WP 2027, with vs without `to_pm180` | values within 5e-7 (run-to-run drift); memory unchanged (that year's peak is not set by Russia) |

Sample of 1,500 production tasks: 0 missing year-parquets; no
`special_region_flag` rows.

**Reproducibility is ~1e-5 relative, not bit-exact.** The unmodified pre-fix
code gave Ethiopia's production value (17,615,218 exposed) in one run and
17,615,114 in a later run on the same inputs (no input file had changed). The
drift is in the float32 population read/sum for very large admin windows
(≈455k valid 100 m pixels), affects one or two large countries per year, and
is far below any meaningful precision. Compare reruns with a relative
tolerance (e.g. 1e-4), not equality.

**The antimeridian fix is a robustness fix with no numeric effect.** Only
pixels with `paf > 0` count, and the PAF raster is clipped to the piece after
`normalize_geom_to_0_360` restores the original 0–360 ring, so the invalid
population mask the old code produced for Russia/Antarctica never changed the
numerator or exposed population. The fix removes the invalid geometry (which
could have raised GEOS errors), not a bias.

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
- A single task-year takes ~2–5 minutes (NI 2027 with 43 admins: ~270 s).

### Failure modes worth knowing

- **Missing stage 2 PAF for a year** — `load_raw_paf_raster` raises
  `FileNotFoundError`. This propagates through `process_single_year` and
  `run_parallel`, failing the task (non-zero exit); the launcher or a Jobmon
  retry resubmits it.
- **Admin-level exception** — logged with traceback; the year is not saved
  and the task exits 1 (see Resume semantics).
- **WindowError on piece × population intersection** — handled inside the
  per-piece loop (a piece's CEA bbox might not overlap the population
  raster); the piece is skipped and the admin's numerator accumulator
  continues with whatever pieces did intersect.
- **Missing admin × year in FHS totals** — `get_population_total` returns
  the sentinel `(1.0, True)` and the row is saved with
  `special_region_flag=True`. The `population_weighted_paf` for those rows
  is effectively `numerator / 1.0`, which is the raw exposed-person-PAF
  sum, not a true population fraction. Downstream code should filter on
  `special_region_flag` before aggregating. (Liechtenstein and Antarctica
  are the admin 0 locations without FHS rows.)
- **Lon-convention drift** — stage 2 PAF rasters for non-NA basins live in
  0-360 lon. The per-piece path explicitly shifts the WGS84 piece to 0-360
  via `normalize_geom_to_0_360` for non-NA basins before clipping the raster.
  The CEA bbox is computed from the piece's intersection with the raster
  bounds, and pyproj wraps longitudes > 180 when reprojecting.
- **Far-north NI exposure (modelling, not code)** — NI PAF reaches 36–55°N
  (Türkiye, the Caucasus, Turkmenistan, southern Russia; tiny PAF, but up to
  20% of Russia's population counted as exposed in NI rows). Input tracks never
  carry ≥ 34 kn north of ~30°N, but decaying segments continue to ~45°N with
  pressure deficits up to 14 hPa, and CLIMADA's default windfield model is
  driven by pressure and translation speed, not the track's reported wind.
  Raise with the hazard-model owners if unintended.
- **Half-pixel registration** inherited from Stages 1–2 (accepted; see
  Stage 1).

---

## Stage 4 — Per-(storm, location) population-weighted exposure

Stage 4 has four pieces, run as two Jobmon workflows:

1. **Part A** (`04_admin_level_exposure_a_main.py`, launched by
   `04_admin_level_exposure_a_launcher.py`): a per-draw metadata pass. For
   every storm × admin polygon that intersects, it records pixel counts,
   affected population (1 km), area and bbox. These numbers size Part B's
   tasks.
2. **Resource assignment** (`04_admin_level_exposure_a_resource_assignment.py`):
   a downstream task in the 4A workflow. It compiles every Part A draw parquet
   into one inventory and bins each (storm × location) task into a
   memory/runtime class for Part B.
3. **Part B** (`04_admin_level_exposure_b_main.py`, launched by
   `04_admin_level_exposure_b_launcher.py`): computes person-storm-hours,
   exposed population and max wind for each (storm × location) and writes a
   one-row parquet.
4. **Post-processing** (`04_admin_level_exposure_b_post_processing.py`): a
   downstream task in the 4B workflow. It reads every expected Part B
   parquet (paths built from the 4A inventory) into one consolidated parquet.

Order: 4A launcher (4A groups → resource assignment) → 4B launcher (4B
groups → post-processing). Both launchers take `ADMIN_LEVEL` (currently
**0**); run the pair once per admin level.

### Provenance of the production outputs

v2 outputs (admin 0): `stage4a_metadata_admin0/` and `stage4b_v2/`, both
built on the grid-stretched Stage 1 rasters, so superseded by v3. Verified
2026-10-08:

- **4A complete:** all 361,200 draws (3,612 combos × 100, baseline excluded)
  have a Stage 1 intensity store and a 4A draw parquet.
- **4B consolidated:** 34,995,927 rows against 34,997,352 tasks in the 4A
  inventory, so **1,425 tasks (0.004%) are missing**, in every basin, with 4A
  estimates up to 31M affected people. Two causes:
  - *Transient failures that were never retried.* Re-running a sample with
    today's code produces output for most of them. The v2 4B `main` had no
    per-task resume, and wrote its group sentinel even when an admin failed.
  - *Grid stretch.* Intensity and exposure were stretched at different steps
    and land-clipped at different shifted positions, so some exposure rasters
    were nearly empty and 4B returned early ("No affected pixels found in
    exposure raster").
- Values are sane: no duplicate keys; exposure is in hours (median 5.5 h per
  exposed person, max 236 h); `max_wind_speed` starts at 17.5 m/s;
  `total_population_exposed` never exceeds `total_population` by more than 3%.
- v2 consolidation dropped `special_region_flag`; v3 keeps it.

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

The ibtracs scripts were **not** part of the 2026-10-08 review, so none of
its fixes (grid, failure handling, antimeridian, population window) have been
ported to `/ibtracs/`.

### Shared conventions (4A and 4B)

- **Reading exposure.** `exposure_hours` carries `units = "hours"`, so a plain
  `xr.open_zarr` decodes it as `timedelta64[ns]`, and `.astype("float32")` then
  gives nanoseconds (28 h → 1.008e14). Both parts open exposure through
  `get_exposure_storm_from_draw(..., decode_times=False)`, which yields float
  hours. Keep it that way; any new reader must pass `decode_times=False` or
  `decode_timedelta=False`.
- **Grid check.** `to_raster(..., expected_res=template_raster.transform.a)`
  raises unless the storm's coords step is exactly 0.1° in both axes. Inside 4A
  and 4B this surfaces as a storm/task failure. Pre-fix (v2) Stage 1 rasters
  are evenly spaced at the wrong step, so v3 Stage 4 refuses them by design.
- **Antimeridian.** The same `normalize_longitudes` (cut at 180 for straddling
  rings) and `to_pm180` (population-window) fixes as Stage 3; see Stage 3
  **Antimeridian handling**. In 4A the population window also sets the
  recorded bbox and `area_100m2`, so it affects resource sizing.
- **Population.** 4A uses the **1 km** grid (a sizing pass); 4B uses **100 m**.
  Both read `world_cylindrical_<m>/<year>q1.tif` from `GRIDED_POP_PATH`.
- **Year.** 4A assigns a storm to every year from its `start_date` year to its
  `end_date` year; 4B uses the `start_date` year only. No sampled storm spans
  two years, so the two agree in practice.

### Part A — Per-(storm, admin) metadata-gathering pass

#### Purpose

Quantify the **work envelope** of each (storm × admin) cell before the heavy
4B compute. For every storm in every year of a draw, record pixel counts,
percent affected, affected population, area and bbox per admin polygon that the
storm's intensity footprint intersects. The output defines **which**
(storm, location) tasks 4B runs and with what memory/runtime.

#### Launcher (`04_admin_level_exposure_a_launcher.py`)

- Task list: `level_4_task_assignments.csv` combos (1965–1969 baseline
  excluded) × draws 0..99 = 361,200 draws.
- Draws are packed **12 per Jobmon task** (`MAX_GROUP_RUNTIME_MIN = 60` /
  `TASK_RUNTIME_MIN = 5`) and run serially inside the task. The grouped task
  list is written to `<SAVE_ROOT>/_workflow_grouped_tasks/grouped_tasks_<uuid>.parquet`.
- Resources: 1 core, 5 G, 60 min, `max_attempts = 5`, retry scaling memory
  ×1.5 and runtime ×3; `max_concurrently_running = 3000`.
- `CHECK_COMPLETION = False` (default): every draw is submitted and each worker
  skips completed draws itself. `True` walks the output tree first.
- Adds the resource-assignment task downstream of every group (it fires
  immediately if there are no groups).

#### CLI

```
python 04_admin_level_exposure_a_main.py \
  --grouped_tasks_parquet <path written by the launcher> \
  --group_id <int> \
  --admin_level <0|1>
```

Exit code is **1** if any draw in the group is incomplete (see Resume and
failure handling).

#### Inputs

| Path | Purpose |
|---|---|
| `stage1_v3/<source>/<variant>/<exp>/<batch_year>/<basin>/intensity/intensity_*.zarr/storm_NNNN/` | Stage 1 storm intensity |
| `stage1_v3/<source>/<variant>/<exp>/<batch_year>/<basin>/exposure_hours/exposure_hours_*.zarr/storm_NNNN/` | Stage 1 storm exposure hours |
| `global_WGS84_admin0.parquet` / `global_WGS84_admin1_0_360.parquet` | Admin polygons for non-NA basins (0–360) |
| `global_WGS84_admin0_normalized.parquet` (admin 0) / `global_WGS84_admin1_0_360.parquet` normalized at load (admin 1) | NA-basin polygons (−180..180) |
| `fhs_population_2023_all_years.parquet` | Loaded for parity with 4B; not used in the 4A output |
| `population-model/results/2026_05_16/world_cylindrical_1000/<year>q1.tif` | Gridded population, **1 km** |

#### Outputs

`SAVE_ROOT = /mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4a_metadata_v3_admin{admin_level}/`

```
<SAVE_ROOT>/<source>/<variant>/<exp>/<batch_year>/<basin>/tc_risk_draw_<draw>/
  storm_metadata/<year>/storm_<storm_id>_<basin>_<source>_<variant>_<exp>_<start>01_<end>12<draw_text>.parquet
  yearly_metadata/<year>/yearly_metadata_<year>_<basin>_<source>_<variant>_<exp>_<start>01_<end>12<draw_text>.parquet
  admin_level_metadata/admin_level_metadata_<basin>_<source>_<variant>_<exp>_<start>01_<end>12<draw_text>.parquet
```

The yearly parquet concatenates the year's storm parquets; the draw parquet
concatenates the yearly ones. Per-row columns (unchanged through both
aggregations):

| Column | Notes |
|---|---|
| `source_id`, `variant_label`, `experiment_id`, `batch_year`, `basin` | Task identifiers |
| `storm_id` | the storm's `storm_id` attr (CLIMADA sid, unpadded int) |
| `year` | int |
| `location_id` | admin polygon's `loc_id` |
| `projection` | CRS of the population raster (ESRI:54034) |
| `resolution` | string label `"estimated 100m from 1km"` |
| `total_population_pixels_100m` | `pop_arr.size × 100`: cells in the masked 1 km population **window** (bbox), scaled to 100 m |
| `affected_population_pixels_100m` | `affected_1km × 100` (cells with pop > 0 and exposure > 0) |
| `percent_affected_100m` | `affected_1km / total_1km × 100` |
| `population_affected` | Σ 1 km population over `pop > 0 & exposure > 0` cells |
| `xmin`, `xmax`, `ymin`, `ymax` | combined CEA bbox of all valid pieces |
| `area_100m2` | sum of the pieces' CEA bbox areas, in **m²** (despite the name); the resource step divides by 10,000 to get 100 m pixels |
| `processing_time_seconds` | per-admin wall time (stringified, two decimals) |

Rows are written only for admins with affected population > 0. Each save
function chmods file + parent to `0o775`.

#### Pipeline flow

Per task (`main`): read the group's rows, skip draws whose draw parquet is
already valid, and run the rest serially (one basin template raster per
basin, cached). Each draw runs in a `try/except`; failures are collected and
the task exits 1 at the end.

Per draw (`process_single_draw`):
1. Skip if `check_if_draw_is_complete` (draw parquet exists, non-empty, ≥ 1 row).
2. Locate the Stage 1 intensity store. If it does not exist the draw had no
   landfalling storms: log and return (no output, not a failure). Then locate
   the exposure store (a missing exposure store raises and becomes a failure).
3. Load the basin-appropriate admin shapes, read storm metadata once, and
   bucket storms by year.
4. For each year (skip if its yearly parquet is complete), for each storm
   (skip if its storm parquet is complete):
   - Open intensity, and exposure via `get_exposure_storm_from_draw`. A missing
     exposure group is a failure.
   - For NA: `normalize_dataset` both. `ensure_min_grid` both.
   - `to_raster(..., expected_res=0.1)` + nearest resample to the basin
     template; `clean_raster` (0 → NaN, crop to the affected bbox). An empty
     raster skips the storm (not a failure).
   - `intersect_shapefile_with_rasters`: admins intersecting intensity > 0.
     No admins skips the storm (not a failure).
   - `reproject_shapefile_to_equal_area`, then per admin (try/except) and per
     antimeridian piece: clip/mask exposure to the piece, bound the 1 km
     population read by the piece's CEA window (`to_pm180`), mask it by the
     piece, resample exposure onto it, and count cells/population.
   - Save the storm parquet **only if every admin succeeded**.
5. Aggregate the year **only if no storm in it failed**.
6. Write the draw parquet **only if nothing in the draw failed**; otherwise
   return the failures.

#### Resume and failure handling

Three layered sentinels, each a parquet that must exist, be non-empty, open
cleanly and have ≥ 1 row: storm (`storm_metadata/<year>/`), year
(`yearly_metadata/<year>/`) and draw (`admin_level_metadata/`). A complete
draw skips the worker; otherwise complete years and storms are skipped.

Failures (missing exposure, a storm-level exception, any admin exception)
keep the corresponding storm, year and draw unsaved, so the draw stays
incomplete. The task finishes the group's other draws, then exits 1. Jobmon
retries it (up to 5 attempts) and the retry recomputes only what is missing.
This matters because **4A defines 4B's task list**: before 2026-10-08 a
failed admin was logged and skipped, the draw was saved anyway, and that
(storm, location) silently never got a 4B task.

Storms that have no affected pixels or no intersecting admin write nothing,
so they are re-examined on every rerun of their draw (cheap).

Verified on the fixed NI test draw (MRI-ESM2-0 r4i1p1f1 ssp585 2025–2029,
draw 53): new vs pre-review code, 40/40 rows identical in every column.
Forcing admin 163 (India) to fail gave exit 1, no yearly or draw parquets and
only the 6 India-free storms saved. A clean rerun skipped those 6, completed
the rest and produced a draw parquet identical to the clean run.

#### Performance characteristics

- **`itertuples(index=False)`** on the admin loop.
- **Per-storm + per-admin `try/except`**, so one bad storm or polygon does not
  kill the group (it is recorded and retried).
- **No explicit `gc.collect()`**: refcount reclamation on `del`.
- **Lazy equal-area bbox**: the per-piece CEA bbox is computed inline by
  intersecting the WGS84 piece with the raster's WGS84 `.bounds` (plus a 0.2°
  buffer) and reprojecting just that box to `ESRI:54034`. Pyproj only ever
  transforms the corners, not every pixel of the intensity + exposure rasters.
  See "Known precision shift" below.
- **Tight chmod** on every saved parquet (file + parent → `0o775`).
- The unused per-admin max-wind pass was removed on 2026-10-08 (4A never
  wrote it).

#### Failure modes worth knowing

- **Pre-fix (v2) Stage 1 inputs**: every stretched storm fails the grid check.
  Rerun Stage 1 first.
- **CEA-boundary edge cases**: `split_antimeridian_geom` (4A copy) splits at
  180 and groups pieces by centroid; see Stage 3. A piece starting exactly at
  180° used to get a globe-wide population window, which inflated `area_100m2`
  for it; `to_pm180` fixes that.

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

Once every 4A group is done, a single downstream Jobmon task compiles the
draw parquets and writes the per-(storm, location) task list that the 4B
launcher consumes.

##### Script

`04_admin_level_exposure_a_resource_assignment.py --admin_level <0|1>`

##### Inputs

| Path | Purpose |
|---|---|
| Every `<SAVE_ROOT>/<source>/<variant>/<exp>/<batch_year>/<basin>/tc_risk_draw_<draw>/admin_level_metadata/*.parquet` | One row per (storm × admin) |
| `/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv` | The combos to walk (× draws 0..99, baseline excluded) |

##### Outputs

```
/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4a_metadata_v3_admin{N}/
  compiled_admin_level_metadata.parquet     # checkpoint: full concat, with a `draw` column (the 4B post-processing inventory)
  resource_estimation_all_storms.parquet    # 4B's task list: one row per (storm × loc × draw)
```

The compiled parquet is a checkpoint so `assign_resources` can be re-run on
its own (e.g. retuning `RESOURCE_MAP`) without redoing the I/O.

##### Constants

| Constant | Value | Notes |
|---|---|---|
| `DRAWS` | `range(100)` | All inner draws |
| `BASELINE_BATCH_YEAR` | `"1965-1969"` | Excluded; the historical baseline isn't run in stage 4 |
| `PIXEL_SIZE_M` / `PIXEL_AREA_M2` | 100 / 10,000 | Converts `area_100m2` (m²) → n 100 m pixels |
| `CHUNK_SIZE` | 1500 | Per-chunk parallel-read size |
| `NUM_CORES` | 40 | Parallel reader pool size |
| `LOC_ID_OVERRIDE_VERY_LARGE` | 22 (admin 0 only) | **Fiji** (the old comment said India; India is 163). Forced into `"very_large"`. The extra memory came from the globe-wide population window that `to_pm180` now fixes; re-evaluate the override after the v3 run. |
| `PIXEL_BINS` | `[0, 1e6, 2e7, 2e8, 5e8, 1.5e9, ∞]` | n_pixels bin edges |
| `SIZE_CLASSES` | `[very_small, small, medium, large, very_large, extreme]` | 6 size classes |
| `RUNTIME_BINS` / `RUNTIME_LABELS` | `[0, 1, 2, 4, 6, ∞]` / `[1, 2, 4, 6, 8]` | Defensive runtime ceiling; the current `RESOURCE_MAP` emits `{1, 2, 4, 6}` only |

##### `RESOURCE_MAP` (v14)

| Size class | Memory | Runtime |
|---|---|---|
| `very_small` | 3 GB | 1 min |
| `small` | 4 GB | 1 min |
| `medium` | 5 GB | 2 min |
| `large` | 10 GB | 2 min |
| `very_large` | 29 GB | 4 min |
| `extreme` | 70 GB | 6 min |

##### Pipeline flow

1. **`enumerate_paths()`**: cross-join the combos with draws 0..99 (baseline
   dropped) into `(parquet_path, draw)` pairs.
2. **`compile_metadata(pairs)`**: `run_parallel` reads them in chunks, stamps
   each row with its draw, concats, and writes the checkpoint. Missing files
   are logged and skipped; any other read error fails the task.
3. **`assign_resources(df)`**: `n_pixels = area_100m2 / 10,000`, bin into size
   classes, map to memory/runtime, apply the loc 22 override and the defensive
   runtime re-bin, drop to one row per
   `(source, variant, exp, batch_year, basin, draw, storm_id, location_id)`,
   and write the launcher-ready parquet.

##### Wiring into the 4A launcher

The 4A launcher creates one `CLIMADA_stage4a_admin{N}_resource_assignment`
task with `upstream_tasks` = every group task (40 cores, 20 G, 60 min,
`max_attempts = 2`, retry scaling memory ×1.5 / runtime ×2). With no groups
to run it fires immediately.

##### Failure modes worth knowing

- **Missing 4A draw parquets** are logged and skipped. Under the v3 failure
  policy a missing draw parquet means that draw failed or had no landfalling
  storms, so check the 4A workflow for failed tasks before running 4B.
- **`area_100m2 > PIXEL_BINS[-1]` (1.5e9)** bins into `"extreme"`; if a real
  task exceeds 70 GB / 6 min, retune `PIXEL_BINS` + `RESOURCE_MAP`.
- **Schema drift**: the step depends on `area_100m2`, `location_id` and the six
  task-identifier columns.

### Part B — Per-(storm, location) compute

#### Purpose

For one `(storm × location_id)`:
1. Load the storm's intensity and exposure-hours rasters from Stage 1.
2. Filter the admin shapefile to the requested `location_id`.
3. Intersect the admin polygon with the storm's intensity > 0 area; if there is
   no intersection, skip.
4. Compute:
   - `max_wind_speed`: the maximum intensity pixel inside the admin polygon
     (rasterize the admin to a label raster on the intensity grid, then take
     the max per label).
   - `person_storm_hours`: Σ over admin pixels of `population (100 m) ×
     exposure_hours`.
   - `total_population_exposed`: Σ population over the same exposed pixels.
   - `total_population`: FHS admin total (full population, not just exposed).
5. Write one row to a per-(storm, loc) parquet.

#### Launcher (`04_admin_level_exposure_b_launcher.py`)

- Task list: `stage4a_metadata_v3_admin{N}/resource_estimation_all_storms.parquet`
  (35.0M tasks in v2 admin 0).
- **Grouping:** tasks are sorted by `(memory_gb, runtime_min_binned)` desc and
  packed into groups of ≤ `MAX_GROUP_RUNTIME_MIN = 360` serial minutes, all in
  one memory class. Each group gets an MD5 `group_hash` of its sorted task
  keys (stable across launches while the task list is unchanged).
- **Resume:** `CHECK_COMPLETION = True` drops groups whose
  `<SAVE_ROOT>/_sentinels/<group_hash>.done` exists. (This replaced the
  earlier per-file bulk walk.)
- **Resources:** memory = the group's class × `MEMORY_MULTIPLIER` (1; set 2
  for a retry launch), runtime = Σ task minutes rounded up to 10, capped at
  360; `max_attempts = 3`, retry scaling memory ×2 / runtime ×1.5;
  `max_concurrently_running = 2500`. Multipliers are applied after grouping,
  so they don't change group hashes.
- Writes the grouped list to
  `stage4a_metadata_v3_admin{N}/_workflow_grouped_tasks_4b/grouped_tasks_<uuid>.parquet`
  and adds the post-processing task downstream of every group.

#### CLI

```
python 04_admin_level_exposure_b_main.py \
  --grouped_tasks_parquet <path written by the launcher> \
  --group_id <int> \
  --admin_level <0|1>
```

Exit code is **1** if any task in the group failed.

#### Inputs

| Path | Purpose |
|---|---|
| `stage1_v3/.../intensity/intensity_*.zarr/storm_NNNN/` | Stage 1 storm intensity |
| `stage1_v3/.../exposure_hours/exposure_hours_*.zarr/storm_NNNN/` | Stage 1 storm exposure hours (read with `decode_times=False`) |
| `global_WGS84_admin0.parquet` / `global_WGS84_admin1_0_360.parquet` | Admin polygons, non-NA basins (0–360) |
| `global_WGS84_admin0_normalized.parquet` / admin 1 normalized at load | NA-basin polygons |
| `fhs_population_2023_all_years.parquet` | FHS admin totals (denominator) |
| `population-model/results/2026_05_16/world_cylindrical_100/<year>q1.tif` | Gridded population, 100 m |

#### Outputs

`SAVE_ROOT` = `stage4b_v3/` (admin 0) or `stage4b_v3_admin1/` (admin 1),
under `/mnt/team/rapidresponse/pub/tropical-storms/climada/output/`.

```
<SAVE_ROOT>/<source>/<variant>/<exp>/<batch_year>/<basin>/tc_risk_draw_<draw>/storm_exposure/<year>/
  storm_<storm_id>_loc_<location_id>_<basin>_<source>_<variant>_<exp>_<start>01_<end>12<draw_text>.parquet
<SAVE_ROOT>/_sentinels/<group_hash>.done
```

`storm_id` in the filename is **unpadded** (e.g. `storm_0_loc_11_…`).

| Column | Notes |
|---|---|
| `draw` | int, the CLIMADA inner draw |
| `storm_id` | int (unpadded) |
| `year` | int, from the storm's `start_date` |
| `location_id` | admin `loc_id` |
| `person_storm_hours` | Σ over exposed pixels of `pop × exposure_hours` |
| `total_population` | FHS admin total for the year (sentinel `1.0` if missing) |
| `total_population_exposed` | Σ pop over exposed pixels |
| `max_wind_speed` | max intensity (m/s) within the admin polygon |
| `special_region_flag` | `True` if `total_population` came from the FHS missing-row sentinel |

The save function chmods file + parent to `0o775`.

#### Pipeline flow

Per task (`main`): for each row of the group, skip it if its output already
exists (`_is_task_complete` checks the exact filename for every year in
`batch_year`), otherwise run `process_single_storm` in a `try/except`.
Failures are collected. If there are none, write the group sentinel;
otherwise print them and exit 1 **without** a sentinel.

`process_single_storm`:
1. Open the storm's intensity group; `year` = `start_date` year.
2. Open exposure via `get_exposure_storm_from_draw`; a missing exposure group
   **raises** (Stage 1 inconsistency).
3. Load the basin-appropriate shapes, filtered to `loc_id == location_id`.
4. For NA: `normalize_dataset` both. Build a minimal storm-extent template
   (`generate_storm_template_raster`, +1° buffer), `ensure_min_grid` both.
5. `to_raster(..., expected_res=0.1)` + nearest resample onto the template;
   `clean_raster` both. An empty raster returns without output.
6. `intersect_shapefile_with_rasters`; no intersection returns without output.
7. Max wind per admin via one `rasterize` label pass.
8. `reproject_shapefile_to_equal_area`, then per admin (try/except): skip if
   its max wind is NaN; split at the antimeridian (4B's
   `split_antimeridian_geom` groups pieces by bbox midpoint and rejects
   wrap-around CEA pieces wider than ~10,000 km); per piece clip/mask
   exposure, bound the 100 m population read by the piece's CEA window
   (`to_pm180`), mask, resample exposure onto it, and accumulate.
9. If any admin raised, **raise** (no partial output); otherwise save the
   one-row parquet.

#### Resume and failure handling

- **Per task:** `_is_task_complete` (exact path, ≥ 1 KB). A Jobmon retry or a
  relaunch reruns the group but skips finished tasks. (The pre-review helper
  looked for a zero-padded storm id that never matches the unpadded
  filenames, and `main` never called it, so every retry redid the whole
  group.)
- **Per group:** the sentinel is written only when every task succeeded, so the
  launcher's sentinel check resubmits any group with a failure.
- **Not failures:** early returns (empty raster, no intersection, NaN max wind)
  write nothing. They are not retried within a run, and are reported as
  missing by post-processing. With the Stage 1 grid fix the "No affected pixels
  found in exposure raster" case should largely disappear.

Verified on the fixed NI test draw: new vs pre-review code, all 40 tasks
produced output with identical values (max relative difference 0).
Forcing location 163 to fail gave exit 1 in every group, no sentinels and
27/40 outputs. A clean rerun skipped the 27 finished tasks, computed only
India's 13, wrote all 4 sentinels, and matched the clean run exactly.

#### Performance characteristics

- **No full-raster reprojection**: the per-piece CEA bbox is a corner
  transform of the WGS84 piece ∩ (raster bounds + 0.2°), shifted to
  −180..180 first (`to_pm180`).
- **Minimal storm template raster** (`generate_storm_template_raster`)
  instead of a basin-wide grid.
- **Vectorized per-admin max wind**: one `rasterio.features.rasterize` pass.
- **Bounded population reads** of the 100 m tiff per piece.
- **Shapefile and FHS table loaded per task**, after the exposure check.

#### Failure modes worth knowing

- **Pre-fix (v2) Stage 1 inputs** fail the grid check; rerun Stage 1 first.
- **Missing intensity store** for a task's draw raises (the task came from 4A,
  so the store should exist); this is recorded as a task failure.
- **FHS missing-row sentinel**: `get_population_total` returns `(1.0, True)`;
  filter on `special_region_flag` before aggregating.
- **Memory**: before `to_pm180`, pieces starting exactly at 180° (Fiji 22,
  Russia 62) read a globe-wide 100 m population strip; this is why loc 22 is
  forced into the 29 GB class.

#### Post-processing downstream task

##### Script

`04_admin_level_exposure_b_post_processing.py --admin_level <0|1>`

##### Inputs

| Path | Purpose |
|---|---|
| `stage4a_metadata_v3_admin{N}/compiled_admin_level_metadata.parquet` | The inventory: one row per expected 4B output, including `year` |
| Every `<SAVE_ROOT>/.../tc_risk_draw_<draw>/storm_exposure/<year>/storm_<storm_id>_loc_<location_id>_*.parquet` | The per-task outputs |

##### Output

```
<SAVE_ROOT>/_consolidated/
  parts/part_NNNNNN.parquet      # one per batch of CHUNK_SIZE × BATCH_CHUNKS inventory rows
  parts_manifest.jsonl           # one JSON line per finished batch: batch_idx, signature, rows, seconds, part_path
  storm_exposure_all.parquet     # the consolidated output
```

Consolidated columns: the 4B columns (with `draw` renamed to
**`tc_risk_draw`**, the inner CLIMADA draw, to avoid confusion with
`storm_draw`), **including `special_region_flag`**, plus
`source_id_variant_label` (`"{source}_{variant}"`), `experiment_id` and
`basin`, stamped from the path. `batch_year` is not included.

##### Constants

| Constant | Value | Notes |
|---|---|---|
| `COMPILED_4A_PARQUET` | `stage4a_metadata_v3_admin{N}/compiled_admin_level_metadata.parquet` | Path source of truth |
| `SAVE_ROOT` | `stage4b_v3/` or `stage4b_v3_admin1/` | Same root the worker writes to |
| `CHUNK_SIZE` | 5000 | Inventory rows per parallel-read chunk |
| `BATCH_CHUNKS` | 64 | Chunks per resumable batch (one part file) |
| `NUM_CORES` | 64 | Parallel reader pool size |
| `WRITE_MASTER_PARQUET` | True | Build `storm_exposure_all.parquet` from the parts |

##### Pipeline flow

1. **`enumerate_meta()`**: read the inventory, projected to the 9 path columns.
2. **`compile_exposure(meta)`**: compute the inventory signature
   (`<inventory file size>-<mtime_ns>-<rows>-<CHUNK_SIZE>-<BATCH_CHUNKS>`) and
   split the inventory into batches. For each batch: if the manifest records it
   as done **with the same signature** and its part file exists, reuse it.
   Otherwise delete any old part for that index, read the batch's 4B parquets
   in parallel (missing files skipped silently, other read errors logged and
   skipped), write the part atomically (`.tmp` → rename) and append a manifest
   line.
3. **`_build_master_from_parts(total_batches)`**: stream parts `1..total_batches`
   (only this inventory's indices) into `storm_exposure_all.parquet`, renaming
   `draw` → `tc_risk_draw`, written atomically.

The signature check was added on 2026-10-08. Previously the manifest was keyed
by batch index alone, so rerunning into an existing `_consolidated/` after 4A
or 4B changed reused **old** parts. In a test with an inventory shrunk from 40
to 15 rows, the old code still returned 40 rows, while the new code
recomputed and returned 15.

##### Wiring into the 4B launcher

One `CLIMADA_stage4b_admin{N}_post_processing` task with `upstream_tasks` =
every group task (40 cores, 100 G, 240 min, `max_attempts = 2`, retry
scaling memory ×1.5 / runtime ×2). It is added unconditionally, so with no
groups to run it fires immediately.

##### Failure modes worth knowing

- **Missing 4B parquets** are skipped. Missing rows mean a 4B early return
  (no output by design) or a task that failed permanently (its group has no
  sentinel; check the 4B workflow). Count `inventory rows − consolidated rows`
  after each run (v2: 1,425).
- **Inventory missing**: `enumerate_meta()` fails fast; rerun the 4A
  resource-assignment step.
- **Final write memory**: the master is streamed part by part (row groups), so
  peak memory is bounded by a part, not the full 35M rows.
