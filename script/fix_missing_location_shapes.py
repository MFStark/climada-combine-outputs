"""One-off: merge special-region geometries (no FHS pop) into their admin 0 parents.

Children listed in SPECIAL_TO_PARENT are unioned into the parent's geometry,
the parent's attributes are kept, and the child rows are dropped from the output.
Standalones (Antarctica, Liechtenstein, Vatican City, Paracel/Spratly Islands)
are intentionally left untouched.
"""
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union


SHP_ROOT_NORMALIZED = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29')
INPUT_PATH = SHP_ROOT_NORMALIZED / "lbd_standard_admin_0.shp"
OUTPUT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile/global_WGS84_admin0_normalized.parquet")

LOC_ID_COL = "loc_id"


SPECIAL_TO_PARENT: dict[int, int] = {
    # United Kingdom (95)
    296: 95, 299: 95, 313: 95, 331: 95, 345: 95, 352: 95, 355: 95, 356: 95,
    368: 95, 382: 95, 415: 95, 421: 95, 60925: 95, 60926: 95, 60927: 95,

    # France (80)
    338: 80, 339: 80, 350: 80, 363: 80, 364: 80, 372: 80, 387: 80, 391: 80,
    394: 80, 395: 80, 423: 80, 60348: 80, 60930: 80,

    # Netherlands (89)
    300: 89, 4641: 89, 4642: 89, 60922: 89,

    # Australia (71)
    318: 71, 319: 71, 375: 71, 60924: 71, 94026: 71, 94027: 71,

    # Norway (90)
    411: 90, 60923: 90,

    # Single-child parents
    297:   79,   # Aland           -> Finland
    311:   92,   # Canary Islands  -> Spain
    332:   78,   # Faroe Islands   -> Denmark
    424:   148,  # Western Sahara  -> Morocco
    60928: 102,  # US Minor Outlying Islands -> United States
    359:   53,   # Kosovo          -> Serbia
    53483: 77,   # Northern Cyprus -> Cyprus
}

# Locations kept standalone (no merge, no drop):
#   60921 Antarctica, 60931 Paracel Islands, 93924 Spratly Islands,
#   360   Liechtenstein, 353 Vatican City


def fix_shapes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    shapefile_ids = set(gdf[LOC_ID_COL])

    missing_children = sorted(c for c in SPECIAL_TO_PARENT if c not in shapefile_ids)
    missing_parents = sorted(
        {p for p in SPECIAL_TO_PARENT.values() if p not in shapefile_ids}
    )
    if missing_children:
        print(f"[warn] {len(missing_children)} child loc_ids not in shapefile: {missing_children}")
    if missing_parents:
        raise ValueError(f"Parent admin 0 loc_ids missing from shapefile: {missing_parents}")

    parent_to_children: dict[int, list[int]] = {}
    for child, parent in SPECIAL_TO_PARENT.items():
        if child in shapefile_ids:
            parent_to_children.setdefault(parent, []).append(child)

    out = gdf.copy()
    for parent, children in parent_to_children.items():
        members = [parent, *children]
        geoms = out.loc[out[LOC_ID_COL].isin(members), "geometry"].tolist()
        merged = unary_union(geoms)
        parent_idx = out.index[out[LOC_ID_COL] == parent][0]
        out.at[parent_idx, "geometry"] = merged
        print(f"[ok] parent {parent}: merged {len(children)} children -> {children}")

    children_to_drop = [c for cs in parent_to_children.values() for c in cs]
    out = out[~out[LOC_ID_COL].isin(children_to_drop)].reset_index(drop=True)

    print(f"[done] rows {len(gdf)} -> {len(out)} (dropped {len(children_to_drop)} children)")
    return out


def main() -> None:
    gdf = gpd.read_file(INPUT_PATH)
    fixed = fix_shapes(gdf)
    fixed.to_parquet(OUTPUT_PATH)
    print(f"[wrote] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
