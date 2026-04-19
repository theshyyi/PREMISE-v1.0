from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import norm, rankdata

from premise.climate_indices.extremes import rx1day, rx5day, sdii as sdii_func

from .case_shared import (
    add_scs_inset,
    align_to_reference,
    basin_mean_series,
    cleanup,
    draw_simple_heatmap,
    extreme_error_scores,
    find_reference_file,
    get_file_list,
    load_china_geoms,
    load_vector,
    make_domain_masks,
    map_common_decor,
    open_product,
    optional_load_vector,
    parse_product_name,
)
from .common import ensure_dir, save_json

DEFAULTS = {
    'file_glob': '*.TIMEFIX.daily.CHINA.nc',
    'reference_keywords': ['CMFDV2', 'CMFD_V2', 'CMFD'],
    'basin_field': 'W1102WB0_2',
    'basins_to_use': ['Haihe River Basin', 'Pearl River Basin', 'Southwest Basin'],
    'china_extent': [73, 136, 18, 54],
    'scs_extent': [105, 125, 3, 25],
    'target_unit': 'mm/day',
    'time_start': None,
    'time_end': None,
    'chunks': {'time': 730, 'lat': 180, 'lon': 180},
    'wet_day_threshold': 1.0,
    'spi_scales': [3, 6, 12],
    'top_products_for_maps': 1,
    'dpi': 300,
}


def _spi_empirical_from_series(series: pd.Series, scale: int) -> pd.Series:
    monthly = series.resample('MS').sum(min_count=1)
    agg = monthly.rolling(scale, min_periods=scale).sum()
    vals = agg.values.astype(float)
    out = np.full_like(vals, np.nan, dtype=float)
    mask = np.isfinite(vals)
    if mask.sum() < max(10, scale + 6):
        return pd.Series(out, index=agg.index)
    ranks = rankdata(vals[mask], method='average')
    probs = (ranks - 0.44) / (len(ranks) + 0.12)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    out[mask] = norm.ppf(probs)
    return pd.Series(out, index=agg.index)


def _corr_rmse(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    df = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    if len(df) < 6:
        return np.nan, np.nan
    av = df['a'].values.astype(float)
    bv = df['b'].values.astype(float)
    corr = np.corrcoef(av, bv)[0, 1] if np.nanstd(av) > 0 and np.nanstd(bv) > 0 else np.nan
    rmse = float(np.sqrt(np.mean((av - bv) ** 2)))
    return corr, rmse


def _extreme_map_triplet(da: xr.DataArray, wet_day_threshold: float):
    rx1 = rx1day(da, freq='YS').mean('time', skipna=True)
    rx5 = rx5day(da, freq='YS').mean('time', skipna=True)
    sdii = sdii_func(da, wet_threshold=wet_day_threshold, freq='YS').mean('time', skipna=True)
    return {'Rx1day': rx1, 'Rx5day': rx5, 'SDII': sdii}


def run_extremes_drought_case_task(task: dict[str, Any]) -> dict[str, Any]:
    cfg = {**DEFAULTS, **task}
    task_name = str(cfg.get('name', 'extremes_drought_case'))
    out_dir = ensure_dir(cfg['output_dir'])
    fig_dir = ensure_dir(out_dir / 'figures')
    tbl_dir = ensure_dir(out_dir / 'tables')

    files = get_file_list(cfg['data_dir'], cfg['file_glob'], cfg.get('files'))
    ref_file = find_reference_file(files, cfg['reference_keywords'])
    product_files = [f for f in files if Path(f) != Path(ref_file)]

    basin_gdf = load_vector(cfg['basin_shp'])
    china_geoms = load_china_geoms(cfg.get('china_shp'))
    islands_gdf = optional_load_vector(cfg.get('islands_shp'))
    nine_dash_gdf = optional_load_vector(cfg.get('nine_dash_shp'))

    ref = open_product(ref_file, time_start=cfg['time_start'], time_end=cfg['time_end'], target_unit=cfg['target_unit'], chunks=cfg['chunks'])
    ref_masks = make_domain_masks(ref, basin_gdf, cfg['basin_field'], cfg['basins_to_use'])

    extreme_rows = []
    spi_rows = []

    for fp in product_files:
        prod_name = parse_product_name(fp)
        try:
            prod = open_product(fp, time_start=cfg['time_start'], time_end=cfg['time_end'], target_unit=cfg['target_unit'], chunks=cfg['chunks'])
            prod_aligned, ref_aligned = align_to_reference(prod, ref)
            if prod_aligned is None:
                continue
            for domain_name, domain_mask in ref_masks.items():
                sim_series = basin_mean_series(prod_aligned, domain_mask)
                obs_series = basin_mean_series(ref_aligned, domain_mask)
                ext = extreme_error_scores(sim_series, obs_series, wet_day_threshold=cfg['wet_day_threshold'])
                extreme_rows.append({'Product': prod_name, 'Domain': domain_name, **ext})
                for scale in cfg['spi_scales']:
                    spi_sim = _spi_empirical_from_series(sim_series, int(scale))
                    spi_obs = _spi_empirical_from_series(obs_series, int(scale))
                    corr, rmse = _corr_rmse(spi_sim, spi_obs)
                    spi_rows.append({'Product': prod_name, 'Domain': domain_name, 'Scale': f'SPI-{scale}', 'Correlation': corr, 'RMSE': rmse})
                cleanup(sim_series, obs_series)
            cleanup(prod, prod_aligned, ref_aligned)
        except Exception as e:
            print(f'[WARN] Failed on {prod_name}: {type(e).__name__}: {e}')
            cleanup()

    extreme_df = pd.DataFrame(extreme_rows)
    spi_df = pd.DataFrame(spi_rows)
    if extreme_df.empty or spi_df.empty:
        raise RuntimeError('No valid extremes/drought diagnostics were computed.')

    extreme_path = tbl_dir / 'extreme_domain_metrics.csv'
    spi_path = tbl_dir / 'spi_domain_metrics.csv'
    extreme_df.to_csv(extreme_path, index=False)
    spi_df.to_csv(spi_path, index=False)

    # determine top products for extreme maps
    top_products = list(extreme_df.groupby('Product')['ExtremeScore'].mean().sort_values(ascending=False).index[: int(cfg['top_products_for_maps'])])

    # Figure 12
    fig = plt.figure(figsize=(15.5, 5.8 * len(top_products)), dpi=int(cfg['dpi']))
    map_names = ['Rx1day', 'Rx5day', 'SDII']
    cmaps = {'Rx1day': 'YlOrRd', 'Rx5day': 'YlOrRd', 'SDII': 'YlGnBu'}
    for i, prod_name in enumerate(top_products):
        prod_file = next(fp for fp in product_files if parse_product_name(fp) == prod_name)
        prod_da = open_product(prod_file, time_start=cfg['time_start'], time_end=cfg['time_end'], target_unit=cfg['target_unit'], chunks=cfg['chunks'])
        prod_aligned, ref_aligned = align_to_reference(prod_da, ref)
        maps = _extreme_map_triplet(prod_aligned, cfg['wet_day_threshold'])
        for j, metric in enumerate(map_names):
            ax = fig.add_axes([0.05 + j * 0.31, 0.12 + (len(top_products)-1-i)*0.40, 0.27, 0.30], projection=ccrs.PlateCarree())
            arr = maps[metric].values
            finite = arr[np.isfinite(arr)]
            vmin = np.nanpercentile(finite, 2) if finite.size else 0.0
            vmax = np.nanpercentile(finite, 98) if finite.size else 1.0
            im = ax.pcolormesh(maps[metric]['lon'].values, maps[metric]['lat'].values, arr, transform=ccrs.PlateCarree(), cmap=cmaps[metric], shading='auto', vmin=vmin, vmax=vmax)
            map_common_decor(ax, china_geoms, islands_gdf, nine_dash_gdf, extent=cfg['china_extent'])
            ax.set_title(f'({chr(97 + i*3 + j)}) {metric}: {prod_name}', fontsize=11, loc='left')
            add_scs_inset(fig, ax, arr, maps[metric]['lon'].values, maps[metric]['lat'].values, china_geoms, islands_gdf, nine_dash_gdf, cmaps[metric], vmin, vmax, cfg['scs_extent'])
            cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.025, ax.get_position().width, 0.012])
            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
            cb.ax.tick_params(labelsize=8)
        cleanup(prod_da, prod_aligned, ref_aligned)
    fig.suptitle('Figure 12. National-scale extreme-index maps for the top-ranked precipitation product(s)', fontsize=15, y=0.98)
    fig12_path = fig_dir / 'Figure12_extreme_maps.png'
    plt.savefig(fig12_path, bbox_inches='tight')
    plt.close(fig)

    # Figure 13
    plot_order = list(extreme_df.groupby('Product')['ExtremeScore'].mean().sort_values(ascending=False).index)
    domains_in_order = [d for d in ['China', *cfg['basins_to_use']] if d in extreme_df['Domain'].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), dpi=int(cfg['dpi']))
    for ax, metric in zip(axes.ravel(), ['Rx1day', 'Rx5day', 'SDII', 'CDD']):
        piv = extreme_df.pivot(index='Product', columns='Domain', values=metric).reindex(index=plot_order, columns=domains_in_order)
        draw_simple_heatmap(ax, piv, metric, cmap='viridis', vmin=0, vmax=1, fmt='.2f', cbar=True, fig=fig)
    fig.suptitle('Figure 13. Extreme-index benchmarking over China and representative basins', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig13_path = fig_dir / 'Figure13_extreme_skill_heatmaps.png'
    plt.savefig(fig13_path, bbox_inches='tight')
    plt.close(fig)

    # Figure 14
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), dpi=int(cfg['dpi']))
    for j, scale in enumerate(cfg['spi_scales']):
        piv_corr = spi_df[spi_df['Scale'] == f'SPI-{scale}'].pivot(index='Product', columns='Domain', values='Correlation').reindex(index=plot_order, columns=domains_in_order)
        draw_simple_heatmap(axes[0, j], piv_corr, f'SPI-{scale} correlation', cmap='viridis', vmin=0, vmax=1, fmt='.2f', cbar=True, fig=fig)
        piv_rmse = spi_df[spi_df['Scale'] == f'SPI-{scale}'].pivot(index='Product', columns='Domain', values='RMSE').reindex(index=plot_order, columns=domains_in_order)
        vmax = np.nanpercentile(piv_rmse.values[np.isfinite(piv_rmse.values)], 98) if np.isfinite(piv_rmse.values).any() else 1.0
        draw_simple_heatmap(axes[1, j], piv_rmse, f'SPI-{scale} RMSE', cmap='viridis', vmin=0, vmax=vmax, fmt='.2f', cbar=True, fig=fig)
    fig.suptitle('Figure 14. Drought-oriented SPI diagnostics over China and representative basins', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig14_path = fig_dir / 'Figure14_spi_drought_heatmaps.png'
    plt.savefig(fig14_path, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'task_name': task_name,
        'reference_file': str(ref_file),
        'product_count': len(product_files),
        'top_products_for_maps': top_products,
        'table_paths': {'extreme_domain_metrics': str(extreme_path), 'spi_domain_metrics': str(spi_path)},
        'figure_paths': {'figure12': str(fig12_path), 'figure13': str(fig13_path), 'figure14': str(fig14_path)},
    }
    save_json(summary, out_dir / f'{task_name}_summary.json')
    return summary


def run_extremes_drought_case_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_extremes_drought_case_task(task) for task in tasks if task.get('enabled', True)]
