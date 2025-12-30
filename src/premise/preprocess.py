# -*- coding: utf-8 -*-

"""
preprocess
=========

Spatial and temporal preprocessing utilities:

- Read shapefiles as GeoDataFrames
- List fields and unique values from shapefiles
- Build regionmask Regions from shapefiles
- Mask/clip xarray datasets or data arrays by shapefile polygons
- Simple lat/lon bounding-box clipping
- Area mean by region (regional aggregation)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import regionmask
import xarray as xr

GeoDataFrame = gpd.GeoDataFrame


# ========== Shapefile utilities ==========

def load_shapefile(shp_path: str) -> GeoDataFrame:
    """
    Read a shapefile into a GeoDataFrame and reproject to WGS84 if needed.
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def list_fields(gdf: GeoDataFrame) -> List[str]:
    """
    List non-geometry fields of a GeoDataFrame.
    """
    return [c for c in gdf.columns if c != gdf.geometry.name]


def unique_values(gdf: GeoDataFrame, field: str) -> List:
    """
    Return sorted unique values of a given field.
    """
    if field not in gdf.columns:
        raise KeyError(f"Field '{field}' not found in GeoDataFrame.")
    return sorted(gdf[field].dropna().unique().tolist())


def create_regions_from_shapefile(
    gdf: GeoDataFrame,
    name_field: Optional[str] = None,
    id_field: Optional[str] = None,
) -> regionmask.Regions:
    """
    Create a regionmask.Regions object from a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon dataset.
    name_field : str or None
        Field name used for region names. If None, use index.
    id_field : str or None
        Field name used for region IDs. If None, use 0..N-1.

    Returns
    -------
    regions : regionmask.Regions
    """
    geometries = gdf.geometry.values

    if name_field is not None:
        if name_field not in gdf.columns:
            raise KeyError(f"name_field '{name_field}' not in GeoDataFrame.")
        names = gdf[name_field].astype(str).tolist()
    else:
        names = [str(i) for i in range(len(gdf))]

    if id_field is not None:
        if id_field not in gdf.columns:
            raise KeyError(f"id_field '{id_field}' not in GeoDataFrame.")
        numbers = gdf[id_field].astype(int).tolist()
    else:
        numbers = list(range(len(gdf)))

    regions = regionmask.Regions(outlines=list(geometries),
                                 names=names,
                                 numbers=numbers)
    return regions


# ========== Spatial masking / clipping ==========

def _get_lon_lat(
    data: Union[xr.Dataset, xr.DataArray],
    lon_name: str = "lon",
    lat_name: str = "lat",
):
    """
    Extract lon/lat coordinates from a dataset or data array.
    """
    if lon_name not in data.coords or lat_name not in data.coords:
        raise KeyError(f"Coordinates '{lon_name}' / '{lat_name}' not found in data.")
    lon = data[lon_name]
    lat = data[lat_name]
    return lon, lat


def clip_to_bbox(
    data: Union[xr.Dataset, xr.DataArray],
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    lon_name: str = "lon",
    lat_name: str = "lat",
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Clip data by a geographic bounding box.

    Returns a subset in lat/lon space.
    """
    lon = data[lon_name]
    lat = data[lat_name]

    if float(lon[0]) <= float(lon[-1]):
        lon_slice = slice(min_lon, max_lon)
    else:
        lon_slice = slice(max_lon, min_lon)

    if float(lat[0]) <= float(lat[-1]):
        lat_slice = slice(min_lat, max_lat)
    else:
        lat_slice = slice(max_lat, min_lat)

    data_clip = data.sel({lon_name: lon_slice, lat_name: lat_slice})
    return data_clip


def mask_with_shapefile(
    data: Union[xr.Dataset, xr.DataArray],
    shp_path: str,
    lon_name: str = "lon",
    lat_name: str = "lat",
    drop: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Mask data using the union of all polygons in a shapefile.

    Keeps grid cells inside any polygon, sets others to NaN.
    """
    gdf = load_shapefile(shp_path)
    regions = create_regions_from_shapefile(gdf)

    lon, lat = _get_lon_lat(data, lon_name=lon_name, lat_name=lat_name)

    mask = regions.mask(lon, lat)  # (lat, lon)
    inside_any = mask.notnull()

    data_masked = data.where(inside_any)

    if drop:
        data_masked = data_masked.sel({lat_name: inside_any.any(dim=lon_name)})
        data_masked = data_masked.sel({lon_name: inside_any.any(dim=lat_name)})

    return data_masked


def mask_by_region_field(
    data: Union[xr.Dataset, xr.DataArray],
    shp_path: str,
    region_field: str,
    region_values: Union[str, Sequence[str]],
    lon_name: str = "lon",
    lat_name: str = "lat",
    drop: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Mask data by a subset of polygons defined by a field in the shapefile.

    Example: keep only certain climate regions or basins.
    """
    gdf = load_shapefile(shp_path)

    if isinstance(region_values, str):
        region_values = [region_values]
    else:
        region_values = list(region_values)

    if region_field not in gdf.columns:
        raise KeyError(f"Region field '{region_field}' not in shapefile.")

    sub_gdf = gdf[gdf[region_field].isin(region_values)].reset_index(drop=True)
    if sub_gdf.empty:
        raise ValueError(f"No geometries found for {region_field} in {region_values}.")

    regions = create_regions_from_shapefile(sub_gdf, name_field=region_field)

    lon, lat = _get_lon_lat(data, lon_name=lon_name, lat_name=lat_name)
    mask = regions.mask(lon, lat)  # (lat, lon)
    inside = mask.notnull()

    data_masked = data.where(inside)

    if drop:
        data_masked = data_masked.sel({lat_name: inside.any(dim=lon_name)})
        data_masked = data_masked.sel({lon_name: inside.any(dim=lat_name)})

    return data_masked


# ========== Regional aggregation (area mean by region) ==========

def area_mean_by_region(
    da: xr.DataArray,
    shp_path: str,
    *,
    region_field: Optional[str] = None,
    lon_name: str = "lon",
    lat_name: str = "lat",
) -> xr.DataArray:
    """
    Compute regional mean time series for each polygon in a shapefile.

    Output is a DataArray with dimension "region" (and "time" if present).

    Parameters
    ----------
    da : xr.DataArray
        Input field, typically with dimensions including lat, lon, and time.
    shp_path : str
        Path to shapefile.
    region_field : str or None
        Name of the field used for region labels. If None, use 0..N-1.
    lon_name, lat_name : str
        Names of lon/lat coordinates.

    Returns
    -------
    da_reg : xr.DataArray
        Regional mean values with dimension "region" (and optionally "time").
    """
    gdf = load_shapefile(shp_path)
    regions = create_regions_from_shapefile(gdf, name_field=region_field)

    lon, lat = _get_lon_lat(da, lon_name=lon_name, lat_name=lat_name)

    mask3d = regions.mask_3D(lon, lat)  # DataArray: region x lat x lon
    region_names = [str(n) for n in regions.names]
    mask3d = mask3d.assign_coords(region=("region", region_names))

    region_means = []

    for i, name in enumerate(region_names):
        mask2d = mask3d.isel(region=i).notnull()  # lat x lon
        da_reg = da.where(mask2d)
        mean_reg = da_reg.mean(dim=(lat_name, lon_name), skipna=True)
        mean_reg = mean_reg.expand_dims({"region": [name]})
        region_means.append(mean_reg)

    da_reg = xr.concat(region_means, dim="region")
    da_reg = da_reg.assign_coords(region=("region", region_names))

    return da_reg
