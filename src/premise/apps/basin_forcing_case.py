from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle, ConnectionPatch

from .case_shared import (
    add_china_outline,
    add_gdf,
    add_scs_inset,
    basin_mean_series,
    build_target_grid,
    cleanup,
    detect_data_var,
    detect_lon_lat,
    draw_target_grid,
    geometry_mask_from_xy,
    get_file_list,
    load_china_geoms,
    load_vector,
    map_common_decor,
    open_product,
    optional_load_vector,
    parse_product_name,
    plot_nine_dash,
    robust_vrange,
    sort_latitude,
    wrap_longitude,
)
from .common import ensure_dir, save_json

DEFAULTS = {
    'file_glob': '*.nc',
    'source_label': 'Source product',
    'basin_field': 'W1102WB0_2',
    'selected_basins': ['Haihe River Basin', 'Pearl River Basin', 'Southwest Basin'],
    'target_basin': 'Haihe River Basin',
    'target_resolution': 0.25,
    'analysis_period': {'start': None, 'end': None},
    'example_period': {'start': None, 'end': None},
    'china_extent': [73, 136, 18, 54],
    'scs_extent': [105, 125, 3, 25],
    'target_unit': 'mm/day',
    'chunks': {'time': 730, 'lat': 180, 'lon': 180},
    'dpi': 300,
    'dem_cmap': 'terrain',
}


def _open_merged_source(files, cfg):
    if len(files) == 1:
        return open_product(files[0], time_start=cfg['analysis_period'].get('start'), time_end=cfg['analysis_period'].get('end'), target_unit=cfg['target_unit'], chunks=cfg['chunks'])
    # merge first as datasets by coords
    ds0 = xr.open_dataset(files[0])
    lon_name, lat_name = detect_lon_lat(ds0)
    ds0 = wrap_longitude(ds0, lon_name)
    lon_name, lat_name = detect_lon_lat(ds0)
    var_name = detect_data_var(ds0, lon_name, lat_name)
    ds0.close()
    ds = xr.open_mfdataset(files, combine='by_coords')
    ds = wrap_longitude(ds, lon_name)
    da = ds[var_name]
    da = sort_latitude(da, lat_name)
    rename_dict = {}
    if lon_name != 'lon':
        rename_dict[lon_name] = 'lon'
    if lat_name != 'lat':
        rename_dict[lat_name] = 'lat'
    if rename_dict:
        da = da.rename(rename_dict)
    if cfg['analysis_period'].get('start') is not None or cfg['analysis_period'].get('end') is not None:
        da = da.sel(time=slice(cfg['analysis_period'].get('start'), cfg['analysis_period'].get('end')))
    return da.astype('float32')


def _load_dem(dem_path):
    ds = xr.open_dataset(dem_path)
    lon_name, lat_name = detect_lon_lat(ds)
    ds = wrap_longitude(ds, lon_name)
    lon_name, lat_name = detect_lon_lat(ds)
    var_name = detect_data_var(ds, lon_name, lat_name)
    da = ds[var_name]
    da = sort_latitude(da, lat_name)
    if lon_name != 'lon' or lat_name != 'lat':
        da = da.rename({lon_name: 'lon', lat_name: 'lat'})
    return da


def _basin_extent(gdf, name, field, pad_ratio=0.05):
    sub = gdf[gdf[field] == name]
    geom = sub.unary_union
    minx, miny, maxx, maxy = sub.total_bounds
    padx = (maxx - minx) * pad_ratio
    pady = (maxy - miny) * pad_ratio
    return geom, [minx - padx, maxx + padx, miny - pady, maxy + pady]


def _save_dataarray_nc(da, path, name='pr'):
    ds = da.to_dataset(name=name)
    ds.to_netcdf(path)
    ds.close()


def run_basin_forcing_case_task(task: dict[str, Any]) -> dict[str, Any]:
    cfg = {**DEFAULTS, **task}
    task_name = str(cfg.get('name', 'basin_forcing_case'))
    out_dir = ensure_dir(cfg['output_dir'])
    fig_dir = ensure_dir(out_dir / 'figures')
    data_dir = ensure_dir(out_dir / 'data')

    basin_gdf = load_vector(cfg['basin_shp'])
    china_geoms = load_china_geoms(cfg.get('china_shp'))
    islands_gdf = optional_load_vector(cfg.get('islands_shp'))
    nine_dash_gdf = optional_load_vector(cfg.get('nine_dash_shp'))
    files = get_file_list(cfg['data_dir'], cfg['file_glob'], cfg.get('files'))
    source_da = _open_merged_source(files, cfg)
    source_name = cfg.get('source_label') or parse_product_name(files[0])

    merged_nc = data_dir / f'{task_name}_merged_source.nc'
    _save_dataarray_nc(source_da, merged_nc)

    # Figure 3: selected basins overview
    if cfg.get('dem_path'):
        dem = _load_dem(cfg['dem_path'])
        fig = plt.figure(figsize=(15.5, 8.5), dpi=int(cfg['dpi']))
        ax_main = fig.add_axes([0.30, 0.16, 0.36, 0.66], projection=ccrs.PlateCarree())
        map_common_decor(ax_main, china_geoms, islands_gdf, nine_dash_gdf, extent=cfg['china_extent'])
        colors = ['#f5c2c7', '#f6d7a7', '#cfe2f3']
        for name, color in zip(cfg['selected_basins'], colors):
            sub = basin_gdf[basin_gdf[cfg['basin_field']] == name]
            add_gdf(ax_main, sub, facecolor=color, edgecolor='black', linewidth=0.7, alpha=0.9)
        ax_main.set_title('Selected basins for basin-oriented forcing preparation', fontsize=12)
        subplot_positions = {
            cfg['selected_basins'][0]: [0.72, 0.60, 0.22, 0.28],
            cfg['selected_basins'][1]: [0.72, 0.10, 0.22, 0.28],
            cfg['selected_basins'][2]: [0.03, 0.10, 0.22, 0.28],
        }
        for name in cfg['selected_basins']:
            geom, ext = _basin_extent(basin_gdf, name, cfg['basin_field'])
            ax = fig.add_axes(subplot_positions[name], projection=ccrs.PlateCarree())
            dem_sub = dem.sel(lon=slice(ext[0], ext[1]), lat=slice(ext[2], ext[3]))
            arr = dem_sub.values
            vmin, vmax = robust_vrange(arr)
            im = ax.pcolormesh(dem_sub['lon'].values, dem_sub['lat'].values, arr, transform=ccrs.PlateCarree(), cmap=cfg['dem_cmap'], shading='auto', vmin=vmin, vmax=vmax)
            ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.8)
            ax.set_extent(ext, crs=ccrs.PlateCarree())
            ax.set_title(name, fontsize=10, pad=4)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7)
        fig3_path = fig_dir / 'Figure3_selected_basins_dem.png'
        plt.savefig(fig3_path, bbox_inches='tight')
        plt.close(fig)
        cleanup(dem)
    else:
        fig3_path = ''

    # Target basin preprocessing products
    target_geom, target_extent = _basin_extent(basin_gdf, cfg['target_basin'], cfg['basin_field'])
    lon2d, lat2d = np.meshgrid(source_da['lon'].values, source_da['lat'].values)
    mask = geometry_mask_from_xy(lon2d, lat2d, target_geom)
    mask_da = xr.DataArray(mask, coords={'lat': source_da['lat'], 'lon': source_da['lon']}, dims=('lat', 'lon'))
    clipped = source_da.where(mask_da)
    lon_new, lat_new = build_target_grid(target_extent, float(cfg['target_resolution']))
    resampled = clipped.interp(lon=lon_new, lat=lat_new, method='linear')
    lon2d_res, lat2d_res = np.meshgrid(resampled['lon'].values, resampled['lat'].values)
    mask_res = geometry_mask_from_xy(lon2d_res, lat2d_res, target_geom)
    resampled = resampled.where(xr.DataArray(mask_res, coords={'lat': resampled['lat'], 'lon': resampled['lon']}, dims=('lat', 'lon')))

    # final subset and save
    start = cfg['analysis_period'].get('start')
    end = cfg['analysis_period'].get('end')
    if start is not None or end is not None:
        resampled = resampled.sel(time=slice(start, end))
    basin_ready_nc = data_dir / f'{task_name}_{cfg["target_basin"].replace(" ", "_")}_basin_ready.nc'
    _save_dataarray_nc(resampled, basin_ready_nc)

    # Figure 6: a-e combined
    fig = plt.figure(figsize=(18, 9), dpi=int(cfg['dpi']))
    ax_a = fig.add_axes([0.04, 0.56, 0.56, 0.34], projection=ccrs.Robinson())
    source_mean = source_da.mean('time', skipna=True)
    vmin, vmax = robust_vrange(source_mean.values)
    im_a = ax_a.pcolormesh(source_mean['lon'].values, source_mean['lat'].values, source_mean.values, transform=ccrs.PlateCarree(), cmap='turbo', shading='auto', vmin=vmin, vmax=vmax)
    # highlight China box
    china_box = Rectangle((cfg['china_extent'][0], cfg['china_extent'][2]), cfg['china_extent'][1]-cfg['china_extent'][0], cfg['china_extent'][3]-cfg['china_extent'][2], transform=ccrs.PlateCarree(), fill=False, edgecolor='red', linewidth=1.2)
    ax_a.add_patch(china_box)
    add_china_outline(ax_a, china_geoms, facecolor='none', edgecolor='black', linewidth=0.6)
    ax_a.set_global(); ax_a.set_title('(a) Domain selection', fontsize=11)
    cax_a = fig.add_axes([0.61, 0.62, 0.01, 0.20]); cb_a = plt.colorbar(im_a, cax=cax_a); cb_a.ax.tick_params(labelsize=8)

    ax_b = fig.add_axes([0.68, 0.58, 0.28, 0.28], projection=ccrs.PlateCarree())
    clip_mean = clipped.mean('time', skipna=True)
    vmin_b, vmax_b = robust_vrange(clip_mean.values)
    im_b = ax_b.pcolormesh(clip_mean['lon'].values, clip_mean['lat'].values, clip_mean.values, transform=ccrs.PlateCarree(), cmap='turbo', shading='auto', vmin=vmin_b, vmax=vmax_b)
    ax_b.add_geometries([target_geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.8)
    ax_b.set_extent(target_extent, crs=ccrs.PlateCarree()); ax_b.set_title('(b) Spatial clipping', fontsize=11)
    fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.02)

    ax_c = fig.add_axes([0.68, 0.20, 0.28, 0.28], projection=ccrs.PlateCarree())
    res_mean = resampled.mean('time', skipna=True)
    vmin_c, vmax_c = robust_vrange(res_mean.values)
    im_c = ax_c.pcolormesh(res_mean['lon'].values, res_mean['lat'].values, res_mean.values, transform=ccrs.PlateCarree(), cmap='turbo', shading='auto', vmin=vmin_c, vmax=vmax_c)
    ax_c.add_geometries([target_geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.8)
    draw_target_grid(ax_c, target_extent, float(cfg['target_resolution']), step=1)
    ax_c.set_extent(target_extent, crs=ccrs.PlateCarree()); ax_c.set_title(f'(c) Resampling to {cfg["target_resolution"]}°', fontsize=11)
    fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.02)

    ts = clipped.mean(dim=('lat', 'lon'), skipna=True).to_series().dropna()
    ex_start = cfg['example_period'].get('start') or str(ts.index.min().date())
    ex_end = cfg['example_period'].get('end') or str(ts.index.max().date())
    ts_ex = ts.loc[ex_start:ex_end]
    monthly = ts_ex.resample('MS').sum() if len(ts_ex) else ts
    annual = monthly.resample('YS').sum()

    ax_d = fig.add_axes([0.04, 0.08, 0.26, 0.30])
    ax_d.plot(monthly.index, monthly.values, linewidth=1.1)
    ax_d.set_title('(d) Temporal aggregation', fontsize=11)
    ax_d.grid(alpha=0.3, linestyle='--')
    ax_d.xaxis.set_major_locator(mdates.YearLocator())
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for label in ax_d.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    ax_e = fig.add_axes([0.34, 0.08, 0.28, 0.30])
    ax_e.bar(annual.index.year.astype(str), annual.values, width=0.7)
    ax_e.set_title('(e) Basin-mean annual aggregation', fontsize=11)
    ax_e.grid(axis='y', alpha=0.3, linestyle='--')

    fig6_path = fig_dir / 'Figure6_preprocessing_chain.png'
    plt.savefig(fig6_path, bbox_inches='tight')
    plt.close(fig)

    # Figure 7: file coverage + merged time series + basin-ready field
    fig = plt.figure(figsize=(14, 8), dpi=int(cfg['dpi']))
    ax1 = fig.add_axes([0.07, 0.67, 0.83, 0.20])
    coverage_records = []
    for i, fp in enumerate(files):
        tmp = xr.open_dataset(fp)
        lon_name, lat_name = detect_lon_lat(tmp)
        var_name = detect_data_var(tmp, lon_name, lat_name)
        dai = tmp[var_name]
        if 'time' in dai.coords and dai.sizes['time'] > 0:
            coverage_records.append({'file': Path(fp).name, 'start': pd.to_datetime(dai['time'].values.min()), 'end': pd.to_datetime(dai['time'].values.max())})
        tmp.close()
    for i, rec in enumerate(coverage_records):
        y = len(coverage_records) - i
        width = rec['end'] - rec['start'] + pd.Timedelta(days=25)
        rect = Rectangle((mdates.date2num(rec['start']), y - 0.35), width.days, 0.6, facecolor='#7FB3D5', edgecolor='black', linewidth=0.6)
        ax1.add_patch(rect)
        ax1.text(mdates.date2num(rec['start']) + 5, y, rec['file'], fontsize=8, va='center', ha='left')
    ax1.set_ylim(0.5, len(coverage_records)+0.8); ax1.set_yticks([]); ax1.grid(axis='x', linestyle='--', alpha=0.3)
    ax1.xaxis_date(); ax1.xaxis.set_major_locator(mdates.YearLocator()); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); ax1.set_title('(a) Input file coverage', fontsize=11, loc='left')

    ax2 = fig.add_axes([0.07, 0.12, 0.43, 0.38])
    ax2.plot(ts.index, ts.values, linewidth=1.2)
    ax2.set_title('(b) Merged monthly basin-mean series', fontsize=11, loc='left')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for label in ax2.get_xticklabels():
        label.set_rotation(45); label.set_ha('right')

    ax3 = fig.add_axes([0.57, 0.16, 0.23, 0.30], projection=ccrs.PlateCarree())
    im3 = ax3.pcolormesh(res_mean['lon'].values, res_mean['lat'].values, res_mean.values, transform=ccrs.PlateCarree(), cmap='turbo', shading='auto')
    ax3.add_geometries([target_geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.8)
    draw_target_grid(ax3, target_extent, float(cfg['target_resolution']), step=1)
    ax3.set_extent(target_extent, crs=ccrs.PlateCarree())
    ax3.set_title('(c) Basin-ready forcing field', fontsize=11, loc='left')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)

    ax4 = fig.add_axes([0.82, 0.16, 0.14, 0.30]); ax4.axis('off')
    txt = f"Output summary\n-------------\nSource: {source_name}\nBasin: {cfg['target_basin']}\nResolution: {cfg['target_resolution']}°\nTime steps: {resampled.sizes.get('time', 0)}\nGrid: {resampled.sizes.get('lat', 0)} x {resampled.sizes.get('lon', 0)}\nRange: {str(pd.to_datetime(resampled['time'].values[0]))[:10]} to {str(pd.to_datetime(resampled['time'].values[-1]))[:10]}\nFile: {basin_ready_nc.name}"
    ax4.text(0, 1, txt, va='top', ha='left', fontsize=9, family='monospace', bbox=dict(boxstyle='round,pad=0.4', facecolor='#F7F7F7', edgecolor='0.6'))

    fig7_path = fig_dir / 'Figure7_basin_ready_product.png'
    plt.savefig(fig7_path, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'task_name': task_name,
        'source_files': [str(f) for f in files],
        'merged_source_nc': str(merged_nc),
        'basin_ready_nc': str(basin_ready_nc),
        'figure_paths': {
            'figure3': str(fig3_path),
            'figure6': str(fig6_path),
            'figure7': str(fig7_path),
        },
    }
    save_json(summary, out_dir / f'{task_name}_summary.json')
    return summary


def run_basin_forcing_case_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_basin_forcing_case_task(task) for task in tasks if task.get('enabled', True)]
