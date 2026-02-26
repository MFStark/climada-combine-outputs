# Script 1: CLIMADA Storm Processing Workflow — Summary

## Purpose:
Processes tropical cyclone (TC) tracks for a given model variant and basin, generating storm hazard, intensity, and exposure outputs at pixel level.

## Outputs:
 1. Raster files of per-storm exposure hours.
 2. Raster files of per-storm yearly maximum intensity.
 3. Saved per-draw storm intensity lists for further analysis.



```
FUNCTION main(source_id, variant_label, experiment_id, batch_year, basin, draw):

    # Step 1: Load tropical cyclone track data
    tc_tracks = prepare_zarr_files(
        source_id, variant_label, experiment_id, batch_year, basin, draw
    )

    # Step 2: Generate spatial centroids for the basin
    centroids = generate_basin_centroids(basin, resolution=RESOLUTION)

    # Step 3: Generate hazard object per track
    hazard = generate_hazard_per_track(tc_tracks, centroids)

    # Step 4: Compute per-storm windspeed (for impact calculations)
    storm_speed_list = generate_intensity_per_storm(hazard, centroids, tc_tracks)

    # Step 5: Compute per-storm exposure
    exposure_list = compute_exposure_per_storm(storm_speed_list, centroids)
    # → Saves per-storm, pixel-level exposure hours as rasters

    # Step 6: Compute per-storm intensity (for impact calculations)
    storm_intensity_list = generate_intensity_per_storm(hazard, centroids, tc_tracks)

    # Step 7: Generate yearly intensity per storm (Max windspeed a pixel experiences over a storm's duration)
    yearly_intensity_list = generate_yearly_intensity_per_storm(hazard, centroids)
    # → Uses CLIMADA's haz.intensity to compute max windspeed per pixel
    # → Saves results as rasters

    # Step 8: Save storm intensity results for this draw
    save_storm_intensity_list_draw(
        storm_intensity_list, source_id, variant_label, experiment_id, batch_year, basin, draw
    )

END FUNCTION
```

# Script 2: Calculate Raw PAFs with impact days effect

## Purpose: 
Compute per-storm and per-year PAFs (Population Attributable Fractions) for each administrative unit.

## Outputs:
Annual PAF per admin unit (annual_admin_paf_df)


## Highlights:
Tracks per-pixel days impacted for multiple storms in a year using a year calendar.
Rasterizes relative risk and clips it to affected pixels and admins.


## Per-Pixel PAF Calculation
The per-pixel Population Attributable Fraction (PAF) is calculated using the formula:
PAF per pixel = (RR - 1) / RR × (days impacted / 365)

Where:
RR is the relative risk for that pixel.
days impacted is the number of days that the pixel was affected by a storm.




## Aggregating PAF to Administrative Units
To get the PAF for an administrative unit, we average the per-pixel PAF values over all pixels within that unit:
PAF per admin = sum(PAF per pixel for all pixels in admin) / number of pixels

## Combining Per-Storm PAFs into Annual PAF
When multiple storms affect the same administrative unit in a year, the annual PAF is computed using the complement formula:
Annual PAF = 1 - product(1 - PAF per storm)

## PAF CODE
                # t_impact = days affected per pixel by this storm
                t_impact = np.asarray(storm_days_impact._ndarray)

                # rr_values = relative risk raster
                rr_values = np.asarray(storm_crop._ndarray)

                # Initialize PAF array
                paf_raw = np.zeros_like(t_impact, dtype=float)

                # Mask valid pixels
                mask = np.isfinite(t_impact) & np.isfinite(rr_values) & (t_impact > 0) & (rr_values != 0)

                # Compute per-pixel PAF for this storm
                paf_raw[mask] = (rr_values[mask] - 1) / rr_values[mask] * (t_impact[mask] / 365)



## Main Function
```
FUNCTION main(
    source_id, variant_label, experiment_id, batch_year, basin, draw,
    relative_risk, sample_name
):

    # Step 0: Load input data
    draw_store = get_draw_zarr_path(source_id, variant_label, experiment_id, batch_year, basin, draw)
    rr_samples_df = load_relative_risk_df(relative_risk)
    shapefile = load_shapefiles()

    # Step 1: Loop over years in the draw
    FOR each year in all_years_in_draw(draw_store):

        # Step 1a: Select a template storm for raster dimensions
        template_storm = get_first_storm_in_year(draw_store, year)

        # Step 1b: Rasterize template storm 
        template_raster = to_raster(template_storm["relative_risk"])

        # Step 1c: Initialize per-year calendar to track pixel-level days impacted
        year_calendar = initialize_year_calendar_like(template_raster, year)

        # Step 2: Loop over all storms in this year
        storms_in_year = get_storms_for_year(draw_store, year)
        storm_count = 0

        FOR each storm_ds in storms_in_year:
            storm_count += 1

            # ------------------------------
            # 2a. Compute per-storm relative risk
            rr_da = generate_relative_risk(storm_ds["relative_risk"], rr_samples_df, sample_name)

            # ------------------------------
            # 2b. Rasterize RR data
            storm_rr = to_raster(rr_da)

            # ------------------------------
            # 2c. Crop RR raster to affected area
            storm_crop = subset_affected_area(storm_rr)

            # ------------------------------
            # 2d. Identify affected administrative units
            affected_admins = intersect_shapefile_with_storm_data(shapefile, rr_da)

            # ------------------------------
            # 2e. Compute per-pixel days impacted by this storm
            storm_days_impact = compute_storm_days_impact(storm_crop, year_calendar, storm_ds.start_date, DEFAULT_DAYS=20)

            # ------------------------------
            # 2f. Compute per-storm, per-admin PAF
            admin_paf_list = []
            FOR each admin in affected_admins:
                paf_raster = compute_pixel_level_paf(storm_days_impact, storm_crop)
                admin_crop = paf_raster.clip(admin.geometry)
                mean_paf = mean(admin_crop)
                admin_paf_list.append({ "admin_id": admin.id, "year": year, "mean_paf": mean_paf })

            admin_paf_df = DataFrame(admin_paf_list)

            # ------------------------------
            # 2g. Accumulate annual PAF per admin
            annual_admin_paf_list = []
            FOR each admin_id, year in admin_paf_df.group_by(["admin_id","year"]):
                annual_paf = 1 - PRODUCT(1 - paf_per_storm)
                annual_admin_paf_list.append({ "admin_id": admin_id, "year": year, "annual_paf": annual_paf })

            annual_admin_paf_df = DataFrame(annual_admin_paf_list)

END FUNCTION
```

# Script 3: Gridded PAF Calculations

## Inputs:
 1. annual_admin_paf_df → per-admin, per-year PAFs from Script 2
 2. 100m population raster → to distribute PAFs spatially
 3. Admin shapefile → to map administrative boundaries

## Outputs:
 1. Per-admin PAF raster

## Logic:
Take annual PAF per admin.
Weight it across the population pixels in that admin.
Save results as rasters for further exposure or impact analysis.

```
FUNCTION main(
    admin_paf_df_path,   # Path to annual_admin_paf_df from Script 2
    population_grid_path, # Path to 100m gridded population
    admin_shapefile_path  # Shapefile for administrative boundaries
):

    # Step 1: Load data
    admin_paf_df = read_csv(admin_paf_df_path)          # Annual PAF per admin
    pop_grid = read_raster(population_grid_path)       # 100m population raster
    admin_shapefile = read_shapefile(admin_shapefile_path)  # Admin geometries

    # Step 2: Loop over administrative units
    FOR each admin in admin_shapefile:

        # 2a: Get annual PAF for this admin
        admin_paf = admin_paf_df.loc[admin.id]

        # 2b: Clip population raster to admin geometry
        admin_pop_grid = clip_raster_to_geometry(pop_grid, admin.geometry)

        # 2c: Compute per-pixel PAF
        # Formula: PAF_pixel = admin_PAF * (population_pixel / total_population_in_admin)
        total_pop = sum(admin_pop_grid)
        paf_grid = admin_paf * (admin_pop_grid / total_pop)

        # 2d: Store or write out per-pixel PAF raster for this admin
        save_raster(paf_grid, filename=f"paf_admin_{admin.id}.tif")

END FUNCTION
```

# Script 4: Max Intensity Per Country

## Inputs:
 1. Max windspeed rasters from Script 1 (per-storm, per-pixel)
 2. Country boundaries (Admin 0 shapefile)

## Outputs:
1. Table/CSV with storm × country maximum windspeed

## Logic:
Overlay storm intensity raster on country polygons.
For each storm, find the maximum windspeed per country.
Aggregate results into a tidy table.

```
FUNCTION main(
    max_intensity_raster_path,  # Max windspeed per storm from Script 1
    admin0_shapefile_path        # Country boundaries
):

    # Step 1: Load data
    intensity_rasters = read_raster_stack(max_intensity_raster_path)  # One raster per storm
    countries = read_shapefile(admin0_shapefile_path)                # Admin0 geometries

    # Step 2: Loop over storms
    FOR each storm_raster in intensity_rasters:

        storm_id = storm_raster.storm_id

        # Step 2a: Loop over countries
        FOR each country in countries:

            # Clip raster to country geometry
            country_raster = clip_raster_to_geometry(storm_raster, country.geometry)

            # Compute max windspeed in this country for this storm
            max_wind = max_value(country_raster)

            # Store results
            record = {
                "storm_id": storm_id,
                "country_id": country.id,
                "max_windspeed": max_wind
            }
            append_to_results(record)

    # Step 3: Save results as table / CSV
    save_csv(results, filename="max_intensity_per_country.csv")

END FUNCTION
```