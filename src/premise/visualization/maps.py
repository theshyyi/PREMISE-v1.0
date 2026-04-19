from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _load_vector(shp_path: str | Path | None):
    if shp_path is None:
        return None
    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def plot_spatial_field(
    da: xr.DataArray,
    *,
    lon_name: str = "lon",
    lat_name: str = "lat",
    shp_path: Optional[str] = None,
    region_field: Optional[str] = None,
    region_values: Optional[Sequence[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    cbar_label: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
    coastlines: bool = True,
    gridlines: bool = True,
):
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

    data_2d = da
    if da.ndim > 2:
        first_dim = [d for d in da.dims if d not in (lat_name, lon_name)][0]
        data_2d = da.isel({first_dim: 0})

    lon = data_2d[lon_name].values
    lat = data_2d[lat_name].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    mesh = ax.pcolormesh(lon2d, lat2d, data_2d.values, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax)

    if coastlines:
        ax.coastlines(resolution="50m", linewidth=0.5)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    gdf = _load_vector(shp_path)
    if gdf is not None:
        if region_field is not None and region_values is not None:
            region_values = [region_values] if isinstance(region_values, str) else list(region_values)
            gdf = gdf[gdf[region_field].isin(region_values)]
        if not gdf.empty:
            feat = ShapelyFeature(gdf.geometry, crs=proj, edgecolor="black", facecolor="none", linewidth=0.8)
            ax.add_feature(feat)

    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.045, pad=0.03)
    if cbar_label:
        cbar.set_label(cbar_label)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_multi_spatial_fields(
    ds: xr.Dataset | dict,
    *,
    variables: Optional[Sequence[str]] = None,
    lon_name: str = "lon",
    lat_name: str = "lat",
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    panel_size: Tuple[float, float] = (5.0, 4.0),
    cmap: str = "viridis",
    shp_path: Optional[str] = None,
    out_path: Optional[str | Path] = None,
):
    if isinstance(ds, dict):
        names = list(variables or ds.keys())
        arrs = {k: ds[k] for k in names}
    else:
        names = list(variables or ds.data_vars)
        arrs = {k: ds[k] for k in names}
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={"projection": proj}, squeeze=False)
    axes_flat = axes.ravel()
    gdf = _load_vector(shp_path)
    for i, name in enumerate(names):
        ax = axes_flat[i]
        da = arrs[name]
        if da.ndim > 2:
            first_dim = [d for d in da.dims if d not in (lat_name, lon_name)][0]
            da = da.isel({first_dim: 0})
        lon2d, lat2d = np.meshgrid(da[lon_name].values, da[lat_name].values)
        mesh = ax.pcolormesh(lon2d, lat2d, da.values, transform=proj, cmap=cmap)
        ax.coastlines(resolution="50m", linewidth=0.4)
        if gdf is not None and not gdf.empty:
            feat = ShapelyFeature(gdf.geometry, crs=proj, edgecolor="black", facecolor="none", linewidth=0.6)
            ax.add_feature(feat)
        ax.set_title(name)
        fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.045, pad=0.03)
    for j in range(n, len(axes_flat)):
        fig.delaxes(axes_flat[j])
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    return fig, axes
