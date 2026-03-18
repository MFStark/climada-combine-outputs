# source /ihme/code/central_comp/miniconda/bin/activate gbd_env # activate the environment through the command line

import os
# Shared Functions
from db_queries import get_location_metadata, get_population
import pandas as pd
import xarray as xr



# Population data from 1970 to 2022
gbd_2021_release_id = 9
fhs_location_set_id = 39
fhs_hierarchy_2021 = get_location_metadata(location_set_id = fhs_location_set_id, release_id = gbd_2021_release_id)
fhs_hierarchy_2021 = fhs_hierarchy_2021[['location_set_id', 'location_id', 'parent_id', 'path_to_top_parent', 'level', 'most_detailed', 'sort_order', 
                         'location_name', 'location_name_short', 'location_type', 'map_id', 'super_region_id', 'super_region_name',
                         'region_id', 'region_name', 'ihme_loc_id', 'local_id', 'lancet_label']]

location_ids = fhs_hierarchy_2021[fhs_hierarchy_2021['level'] <= 3]['location_id'].tolist()
# Get population
all_population = get_population(
    age_group_id=22,
    release_id=gbd_2021_release_id,
    year_id=list(range(1970, 2101)),
    location_id=location_ids,
    sex_id=3
)
all_population = all_population[["age_group_id", "location_id", "year_id", "sex_id", "population"]]
# write all_population to a parquet file
all_population.to_parquet("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2021.parquet", index=False)

# Future population data from 2021 to 2100


pop_future = xr.open_dataset("/mnt/share/forecasting/data/9/future/population/20250219_draining_fix_old_pop_v5/population.nc")

# pop_future: dims = draw:100, scenario:1, location_id:482, year_id:80, age_group_id:25, sex_id:2
# data var = 'population'

# 1. Aggregate across sex -> sex_id = 3
pop_sex_agg = pop_future['population'].sum(dim='sex_id').assign_coords(sex_id=3)

# 2. Aggregate across age_group -> age_group_id = 22
pop_agg = pop_sex_agg.sum(dim='age_group_id').assign_coords(age_group_id=22)

# 3. Drop years 2021 and 2022
pop_agg = pop_agg.sel(year_id=~pop_agg.year_id.isin([2021, 2022]))

# 4. Take mean across draws
pop_mean = pop_agg.mean(dim='draw')

# 5. Convert to dataframe
df = pop_mean.to_dataframe(name='population').reset_index()

# 6. Keep only relevant columns
df = df[['location_id', 'year_id', 'age_group_id', 'sex_id', 'population']]

df.to_parquet("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2021_future.parquet")


# Combine all
combined_pop = pd.concat([all_population, df])
combined_pop.to_parquet("/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2021_all_years.parquet")
