from __future__ import annotations

import geopandas as gpd

def load_shapefile(shp_path: str):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf
