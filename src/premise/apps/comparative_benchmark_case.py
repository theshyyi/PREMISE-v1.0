from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs

from .case_shared import (
    add_scs_inset,
    align_to_reference,
    basin_mean_series,
    cleanup,
    continuous_metrics_series,
    draw_simple_heatmap,
    entropy_weight_topsis,
    event_metrics_series,
    extreme_error_scores,
    find_reference_file,
    get_file_list,
    grid_metrics,
    load_china_geoms,
    load_vector,
    make_domain_masks,
    map_common_decor,
    normalize_metric_for_topsis,
    open_product,
    parse_product_name,
    plot_taylor,
    optional_load_vector,
    robust_vrange,
    taylor_stats,
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
    'min_valid_days': 365,
    'event_threshold_china': 3.0,
    'event_threshold_basin': 10.0,
    'top_n_map_products': 3,
    'time_start': None,
    'time_end': None,
    'chunks': {'time': 730, 'lat': 180, 'lon': 180},
    'dpi': 300,
}


def _metric_meta():
    return {
        'Bias': ('RdBu_r', None),
        'RMSE': ('viridis', (0, None)),
        'CC': ('viridis', (0, 1)),
        'KGE': ('viridis', None),
    }


def run_comparative_benchmark_case_task(task: dict[str, Any]) -> dict[str, Any]:
    cfg = {**DEFAULTS, **task}
    task_name = str(cfg.get('name', 'comparative_benchmark_case'))
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

    cont_rows, event_rows, extreme_rows, taylor_rows = [], [], [], []

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
                cont = continuous_metrics_series(sim_series, obs_series)
                threshold = cfg['event_threshold_china'] if domain_name == 'China' else cfg['event_threshold_basin']
                evt = event_metrics_series(sim_series, obs_series, threshold=threshold)
                ext = extreme_error_scores(sim_series, obs_series)
                cont_rows.append({'Product': prod_name, 'Domain': domain_name, **cont})
                event_rows.append({'Product': prod_name, 'Domain': domain_name, **evt})
                extreme_rows.append({'Product': prod_name, 'Domain': domain_name, **ext})
                if domain_name == 'China':
                    std_ratio, corr = taylor_stats(sim_series, obs_series)
                    taylor_rows.append({'Product': prod_name, 'StdRatio': std_ratio, 'Correlation': corr})
                cleanup(sim_series, obs_series)
            cleanup(prod, prod_aligned, ref_aligned)
        except Exception as e:
            print(f'[WARN] Failed on {prod_name}: {type(e).__name__}: {e}')
            cleanup()

    cont_df = pd.DataFrame(cont_rows)
    event_df = pd.DataFrame(event_rows)
    extreme_df = pd.DataFrame(extreme_rows)
    taylor_df = pd.DataFrame(taylor_rows).set_index('Product') if taylor_rows else pd.DataFrame(columns=['StdRatio', 'Correlation'])

    if cont_df.empty:
        raise RuntimeError('No valid product metrics were computed.')

    cont_path = tbl_dir / 'continuous_metrics.csv'
    event_path = tbl_dir / 'event_metrics.csv'
    extreme_path = tbl_dir / 'extreme_scores.csv'
    cont_df.to_csv(cont_path, index=False)
    event_df.to_csv(event_path, index=False)
    extreme_df.to_csv(extreme_path, index=False)

    merge_df = cont_df.merge(event_df[['Product', 'Domain', 'EventScore']], on=['Product', 'Domain'], how='left')
    merge_df = merge_df.merge(extreme_df[['Product', 'Domain', 'ExtremeScore']], on=['Product', 'Domain'], how='left')
    merge_df['AbsBias'] = merge_df['Bias'].abs()

    criteria_meta = {
        'AbsBias': False,
        'RMSE': False,
        'CC': True,
        'KGE': True,
        'EventScore': True,
        'ExtremeScore': True,
    }
    mcdm_scores = {}
    for domain in merge_df['Domain'].unique():
        sub = merge_df[merge_df['Domain'] == domain].copy().set_index('Product')
        decision = pd.DataFrame(index=sub.index)
        for metric, higher_better in criteria_meta.items():
            decision[metric] = normalize_metric_for_topsis(sub[metric].values, higher_better=higher_better)
        score, _ = entropy_weight_topsis(decision)
        mcdm_scores[domain] = score
    mcdm_df = pd.DataFrame(mcdm_scores).sort_values(by='China', ascending=False)
    mcdm_path = tbl_dir / 'mcdm_scores.csv'
    mcdm_df.to_csv(mcdm_path)

    plot_order = list(mcdm_df.index)
    domains_in_order = [d for d in ['China', *cfg['basins_to_use']] if d in cont_df['Domain'].unique()]

    # Figure 8: top-N maps
    top_products = plot_order[: int(cfg['top_n_map_products'])]
    fig = plt.figure(figsize=(16.5, 4.2 * len(top_products)), dpi=int(cfg['dpi']))
    left0, bottom0, w, h, xgap, ygap = 0.05, 0.08, 0.25, 0.24, 0.05, 0.06
    for i, prod_name in enumerate(top_products):
        prod_file = next(fp for fp in product_files if parse_product_name(fp) == prod_name)
        prod_da = open_product(prod_file, time_start=cfg['time_start'], time_end=cfg['time_end'], target_unit=cfg['target_unit'], chunks=cfg['chunks'])
        prod_aligned, ref_aligned = align_to_reference(prod_da, ref)
        prod_month = prod_aligned.resample(time='MS').sum()
        ref_month = ref_aligned.resample(time='MS').sum()
        bias_map, rmse_map, _, kge_map = grid_metrics(prod_month, ref_month, min_valid_days=max(12, int(cfg['min_valid_days'] // 30)))
        arrays = [bias_map.values, rmse_map.values, kge_map.values]
        finite_bias = arrays[0][np.isfinite(arrays[0])]
        finite_rmse = arrays[1][np.isfinite(arrays[1])]
        finite_kge = arrays[2][np.isfinite(arrays[2])]
        bias_lim = np.nanpercentile(np.abs(finite_bias), 98) if finite_bias.size else 1.0
        rmse_lim = np.nanpercentile(finite_rmse, 98) if finite_rmse.size else 1.0
        kge_min = np.nanpercentile(finite_kge, 2) if finite_kge.size else -0.5
        kge_max = np.nanpercentile(finite_kge, 98) if finite_kge.size else 1.0
        cmaps = ['RdBu_r', 'viridis', 'YlGnBu']
        vmins = [-bias_lim, 0.0, kge_min]
        vmaxs = [bias_lim, rmse_lim, kge_max]
        titles = [f'({chr(97 + i*3)}) Bias: {prod_name}', f'({chr(98 + i*3)}) RMSE: {prod_name}', f'({chr(99 + i*3)}) KGE: {prod_name}']
        for j in range(3):
            ax = fig.add_axes([left0 + j * (w + xgap), bottom0 + (len(top_products)-1-i)*(h+ygap), w, h], projection=ccrs.PlateCarree())
            im = ax.pcolormesh(ref_month['lon'].values, ref_month['lat'].values, arrays[j], transform=ccrs.PlateCarree(), cmap=cmaps[j], shading='auto', vmin=vmins[j], vmax=vmaxs[j])
            map_common_decor(ax, china_geoms, islands_gdf, nine_dash_gdf, extent=cfg['china_extent'])
            ax.set_title(titles[j], fontsize=11, loc='left')
            add_scs_inset(fig, ax, arrays[j], ref_month['lon'].values, ref_month['lat'].values, china_geoms, islands_gdf, nine_dash_gdf, cmaps[j], vmins[j], vmaxs[j], cfg['scs_extent'])
            cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.025, ax.get_position().width, 0.012])
            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
            cb.ax.tick_params(labelsize=8)
        cleanup(prod_da, prod_aligned, ref_aligned, prod_month, ref_month, bias_map, rmse_map, kge_map)
    fig.suptitle('Figure 8. National-scale benchmarking maps for the top-ranked precipitation products', fontsize=15, y=0.98)
    fig8_path = fig_dir / 'Figure8_top_products_national_maps.png'
    plt.savefig(fig8_path, bbox_inches='tight')
    plt.close(fig)

    # Figure 9
    cont_ordered = cont_df.copy()
    cont_ordered['Product'] = pd.Categorical(cont_ordered['Product'], categories=plot_order, ordered=True)
    cont_ordered = cont_ordered.sort_values('Product')
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), dpi=int(cfg['dpi']))
    for ax, metric in zip(axes.ravel(), ['Bias', 'RMSE', 'CC', 'KGE']):
        piv = cont_ordered.pivot(index='Product', columns='Domain', values=metric).reindex(index=plot_order, columns=domains_in_order)
        if metric == 'Bias':
            vmax = np.nanpercentile(np.abs(piv.values[np.isfinite(piv.values)]), 98) if np.isfinite(piv.values).any() else 1.0
            vmin, cmap = -vmax, 'RdBu_r'
        elif metric == 'RMSE':
            vmin, vmax, cmap = 0.0, np.nanpercentile(piv.values[np.isfinite(piv.values)], 98) if np.isfinite(piv.values).any() else 1.0, 'viridis'
        elif metric == 'CC':
            vmin, vmax, cmap = 0.0, 1.0, 'viridis'
        else:
            finite = piv.values[np.isfinite(piv.values)]
            vmin = np.nanpercentile(finite, 2) if finite.size else -0.5
            vmax = np.nanpercentile(finite, 98) if finite.size else 1.0
            cmap = 'viridis'
        draw_simple_heatmap(ax, piv, metric, cmap=cmap, vmin=vmin, vmax=vmax, fmt='.2f', cbar=True, fig=fig)
    fig.suptitle('Figure 9. Continuous-metric comparison over China and representative basins', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig9_path = fig_dir / 'Figure9_continuous_heatmaps.png'
    plt.savefig(fig9_path, bbox_inches='tight')
    plt.close(fig)

    # Figure 10
    event_score_piv = event_df.pivot(index='Product', columns='Domain', values='EventScore').reindex(index=plot_order, columns=domains_in_order)
    extreme_score_piv = extreme_df.pivot(index='Product', columns='Domain', values='ExtremeScore').reindex(index=plot_order, columns=domains_in_order)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.8), dpi=int(cfg['dpi']))
    draw_simple_heatmap(axes[0], event_score_piv, f'(a) Event score (POD/FAR/CSI; China = {cfg["event_threshold_china"]:g}, basins = {cfg["event_threshold_basin"]:g} mm/day)', cmap='YlGnBu', vmin=0, vmax=1, fmt='.2f', cbar=True, fig=fig)
    draw_simple_heatmap(axes[1], extreme_score_piv, '(b) Extreme-index score (Rx1day, Rx5day, SDII, CDD)', cmap='YlGnBu', vmin=0, vmax=1, fmt='.2f', cbar=True, fig=fig)
    fig.suptitle('Figure 10. Event and extreme benchmarking over China and representative basins', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig10_path = fig_dir / 'Figure10_event_extreme_heatmaps.png'
    plt.savefig(fig10_path, bbox_inches='tight')
    plt.close(fig)

    # Figure 11
    base_colors = plt.cm.tab20(np.linspace(0, 1, max(len(plot_order), 3)))
    color_map = {name: base_colors[i % len(base_colors)] for i, name in enumerate(plot_order)}
    fig = plt.figure(figsize=(13.0, 7.5), dpi=int(cfg['dpi']))
    ax_taylor = fig.add_axes([0.05, 0.12, 0.40, 0.78], projection='polar')
    handles = plot_taylor(ax_taylor, taylor_df.reindex(plot_order), color_map)
    ax_rank = fig.add_axes([0.56, 0.16, 0.38, 0.70])
    draw_simple_heatmap(ax_rank, mcdm_df.reindex(index=plot_order, columns=domains_in_order), 'MCDM ranking score (entropy-weighted TOPSIS)', cmap='YlGnBu', vmin=0, vmax=1, fmt='.3f', cbar=True, fig=fig)
    fig.legend(handles=handles[:min(len(handles), 12)], loc='lower left', bbox_to_anchor=(0.05, 0.01), ncol=3, fontsize=9, frameon=False)
    fig.suptitle('Figure 11. Taylor diagnosis and integrated MCDM ranking', fontsize=14, y=0.97)
    fig11_path = fig_dir / 'Figure11_taylor_mcdm.png'
    plt.savefig(fig11_path, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'task_name': task_name,
        'reference_file': str(ref_file),
        'product_count': len(product_files),
        'top_products': top_products,
        'table_paths': {
            'continuous_metrics': str(cont_path),
            'event_metrics': str(event_path),
            'extreme_scores': str(extreme_path),
            'mcdm_scores': str(mcdm_path),
        },
        'figure_paths': {
            'figure8': str(fig8_path),
            'figure9': str(fig9_path),
            'figure10': str(fig10_path),
            'figure11': str(fig11_path),
        },
    }
    save_json(summary, out_dir / f'{task_name}_summary.json')
    return summary


def run_comparative_benchmark_case_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_comparative_benchmark_case_task(task) for task in tasks if task.get('enabled', True)]
