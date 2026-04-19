from __future__ import annotations

from pathlib import Path
from typing import Sequence

import geopandas as gpd
import regionmask
import xarray as xr


def _ensure_geo_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def _subset_axis(data: xr.Dataset | xr.DataArray, coord_name: str, vmin: float, vmax: float):
    coord = data[coord_name]
    if float(coord[0]) <= float(coord[-1]):
        return data.sel({coord_name: slice(vmin, vmax)})
    return data.sel({coord_name: slice(vmax, vmin)})


def clip_to_bbox(
    data: xr.Dataset | xr.DataArray,
    *,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    lon_name: str = 'lon',
    lat_name: str = 'lat',
) -> xr.Dataset | xr.DataArray:
    out = _subset_axis(data, lon_name, min_lon, max_lon)
    out = _subset_axis(out, lat_name, min_lat, max_lat)
    return out


def clip_with_vector(
    data: xr.Dataset | xr.DataArray,
    *,
    vector_path: str | Path,
    lon_name: str = 'lon',
    lat_name: str = 'lat',
    region_field: str | None = None,
    region_values: str | Sequence[str] | None = None,
    drop: bool = True,
) -> xr.Dataset | xr.DataArray:
    vector_path = Path(vector_path)
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector file not found: {vector_path}")

    gdf = gpd.read_file(vector_path)
    gdf = _ensure_geo_crs(gdf)

    if region_field is not None and region_values is not None:
        if region_field not in gdf.columns:
            raise KeyError(f"Field '{region_field}' not found in {vector_path}")
        if isinstance(region_values, str):
            region_values = [region_values]
        gdf = gdf[gdf[region_field].isin(list(region_values))].reset_index(drop=True)
        if gdf.empty:
            raise ValueError(f"No geometries matched {region_field}={region_values}")

    regions = regionmask.from_geopandas(gdf)
    mask = regions.mask(data[lon_name], data[lat_name]).notnull()
    out = data.where(mask)

    if drop:
        out = out.sel({lat_name: mask.any(dim=lon_name)})
        out = out.sel({lon_name: mask.any(dim=lat_name)})

    return out
