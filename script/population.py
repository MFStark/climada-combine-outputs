# source /ihme/code/central_comp/miniconda/bin/activate gbd_env # activate the environment through the command line

import os
# Shared Functions
from db_queries import get_location_metadata, get_population
import pandas as pd
import xarray as xr



# Population data from 1970 to 2022
gbd_2023_release_id = 16
fhs_location_set_id = 39
fhs_hierarchy_2023 = get_location_metadata(location_set_id = fhs_location_set_id, release_id = gbd_2023_release_id)
fhs_hierarchy_2023 = fhs_hierarchy_2023[['location_set_id', 'location_id', 'parent_id', 'path_to_top_parent', 'level', 'most_detailed', 'sort_order', 
                         'location_name', 'location_name_short', 'location_type', 'map_id', 'super_region_id', 'super_region_name',
                         'region_id', 'region_name', 'ihme_loc_id', 'local_id', 'lancet_label']]

location_ids = fhs_hierarchy_2023[fhs_hierarchy_2023['level'] <= 4]['location_id'].tolist()
# Get population
all_population = get_population(
    age_group_id=22,
    release_id=gbd_2023_release_id,
    year_id=list(range(1970, 2101)),
    location_id=location_ids,
    sex_id=3
)
all_population = all_population[["age_group_id", "location_id", "year_id", "sex_id", "population"]]
# write all_population to a parquet file
all_population.to_parquet("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2023.parquet", index=False)

# Future population data from 2023 to 2100


pop_future = xr.open_dataset("/mnt/share/forecasting/data/32/future/population/future_population_s130v41/population_agg.nc")

# 1. Select sex_id 3 and age_group_id 22
pop_agg = pop_future.sel(sex_id=3, age_group_id=22)

# 2. Take mean across draws
pop_mean = pop_agg["draws"].mean(dim='draw')

# 3. Convert to dataframe
df = pop_mean.to_dataframe(name='population').reset_index()

# 4. Keep only relevant columns
df = df[['location_id', 'year_id', 'age_group_id', 'sex_id', 'population']]

# 5. Drop years 2023 and 2024, since we have actual population data for those years
df = df[~df['year_id'].isin([2023, 2024])]

df.to_parquet("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2023_future.parquet")


# Combine all
combined_pop = pd.concat([all_population, df])
combined_pop.to_parquet("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2023_all_years.parquet")
