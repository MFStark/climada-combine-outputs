"""One-off: regenerate 0-360-longitude admin shapefiles from the LBD source.

Reads `lbd_standard_admin_{0,1,2}.shp` from the SNFS LBD shapefile root,
shifts each polygon from the EPSG:4326 [-180, 180] convention to [0, 360]
(matching CLIMADA's basin centroid convention), and writes one parquet
per admin level to the team mount.

Polygon handling:
- Entirely in [0, 180]: passes through unchanged.
- Entirely in [-180, 0]: shifted +360.
- Crossing the prime meridian normally: split at lon=0, west half shifted +360.
- Crossing the antimeridian (touching both ±180): clipped at lon=0, the
  west half shifted +360, then re-joined with a small buffer to close the
  numerical seam.
"""
from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
import pandas as pd

INPUT_ROOT = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29')
OUTPUT_ROOT = Path('/mnt/team/rapidresponse/pub/tropical-storms/data/global_shapefile')

ADMIN_LEVELS = [0, 1]
SIMPLIFIED_SUFFIX = ''

# Default writes alongside the existing canonical files with a "_0_360" suffix so
# nothing is clobbered. To replace `global_WGS84_admin{N}.parquet` directly,
# change this to "global_WGS84_admin{admin_level}.parquet".
OUTPUT_FILENAME_TEMPLATE = "global_WGS84_admin{admin_level}_0_360.parquet"

LEFT_BOX = box(-180, -90, 0, 90)
RIGHT_BOX = box(0, -90, 180, 90)
BUFFER = 1e-4

FHS_LOC_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/fhs_hierarchy_39_32.csv")

def filter_to_fhs_admin1(gdf: gpd.GeoDataFrame, fhs_path: Path = FHS_LOC_PATH) -> gpd.GeoDataFrame:
    """Filter a GeoDataFrame to only include admin1 regions present in the FHS hierarchy."""
    fhs = pd.read_csv(fhs_path)
    admin1_fhs_ids = fhs[fhs["level"] == 4]["location_id"].unique()
    filtered_gdf = gdf[gdf["loc_id"].isin(admin1_fhs_ids)].copy()
    return filtered_gdf

def shift_left_piece(geom):
    """Shift a piece clipped to [-180, 0] into [180, 360].

    Uses `<= 0` so a coord exactly on the seam (x == 0) shifts to 360,
    keeping the west half of a split polygon flush with the eastern edge.
    """
    def shift_coords(coords):
        return [(x + 360 if x <= 0 else x, y) for x, y in coords]
    def shift_poly(poly):
        return Polygon(
            shift_coords(poly.exterior.coords),
            [shift_coords(r.coords) for r in poly.interiors],
        )
    if geom.geom_type == 'Polygon':
        return shift_poly(geom)
    if geom.geom_type == 'MultiPolygon':
        return MultiPolygon([shift_poly(p) for p in geom.geoms])
    return geom


def shift_normal(geom):
    """Shift a geometry entirely in negative space (maxx <= 0) into positive space."""
    def shift_coords(coords):
        return [(x + 360 if x < 0 else x, y) for x, y in coords]
    def shift_poly(poly):
        return Polygon(
            shift_coords(poly.exterior.coords),
            [shift_coords(r.coords) for r in poly.interiors],
        )
    if geom.geom_type == 'Polygon':
        return shift_poly(geom)
    if geom.geom_type == 'MultiPolygon':
        return MultiPolygon([shift_poly(p) for p in geom.geoms])
    return geom


def reproject_geometry(geom):
    minx, _, maxx, _ = geom.bounds

    if minx >= 0:
        return geom
    if maxx <= 0:
        return shift_normal(geom)

    parts = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)

    result_parts = []
    for part in parts:
        pminx, _, pmaxx, _ = part.bounds
        if pminx < -180 + BUFFER and pmaxx > 180 - BUFFER:
            left = part.intersection(LEFT_BOX)
            right = part.intersection(RIGHT_BOX)
            left_shifted = shift_left_piece(left)
            combined = unary_union(
                [left_shifted.buffer(BUFFER), right.buffer(BUFFER)]
            ).buffer(-BUFFER)
            result_parts.append(combined)
        elif pminx < -180 + BUFFER:
            result_parts.append(shift_normal(part))
        elif pmaxx > 180 - BUFFER:
            result_parts.append(part)
        elif pmaxx <= 0:
            result_parts.append(shift_normal(part))
        elif pminx >= 0:
            result_parts.append(part)
        else:
            left = part.intersection(LEFT_BOX)
            right = part.intersection(RIGHT_BOX)
            result_parts.append(shift_left_piece(left))
            result_parts.append(right)

    return unary_union(result_parts)


def process_admin_level(admin_level: int) -> None:
    shp_path = INPUT_ROOT / f"lbd_standard_admin_{admin_level}{SIMPLIFIED_SUFFIX}.shp"
    out_path = OUTPUT_ROOT / OUTPUT_FILENAME_TEMPLATE.format(admin_level=admin_level)

    print(f"[admin {admin_level}] reading {shp_path}")
    gdf = gpd.read_file(shp_path)

    if admin_level == 1:
        print(f"[admin {admin_level}] filtering to FHS admin1 regions")
        gdf = filter_to_fhs_admin1(gdf)

    print(f"[admin {admin_level}] reprojecting {len(gdf)} rows to 0-360")
    gdf['geometry'] = gdf['geometry'].apply(reproject_geometry)

    print(f"[admin {admin_level}] writing {out_path}")
    gdf.to_parquet(out_path)
    print(f"[admin {admin_level}] done")


def main() -> None:
    for admin_level in ADMIN_LEVELS:
        process_admin_level(admin_level)


if __name__ == '__main__':
    main()
