from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from premise.product_evaluation.pipeline import run_tasks as run_evaluation_tasks
from premise.visualization.distributions import plot_metric_heatmap

from .common import ensure_dir, save_json


def _combine_metric_tables(csv_paths: list[Path], product_names: list[str], out_csv: Path) -> pd.DataFrame:
    rows = []
    for product, path in zip(product_names, csv_paths):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row['product'] = product
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        cols = ['product'] + [c for c in out.columns if c != 'product']
        out = out[cols]
        out.to_csv(out_csv, index=False)
    return out


def _write_markdown(report: dict[str, Any], out_path: Path) -> None:
    lines = ['# Dataset evaluator report', '']
    lines.append(f"- task_name: {report.get('task_name', '')}")
    lines.append(f"- mode: {report.get('mode', '')}")
    lines.append(f"- output_dir: {report.get('output_dir', '')}")
    lines.append('')
    lines.append('## Products')
    for rec in report.get('records', []):
        lines.append(f"- {rec.get('name', '')}: {rec.get('status', '')}")
        if rec.get('error_message'):
            lines.append(f"  - error: {rec['error_message']}")
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def run_dataset_evaluator_task(task: dict[str, Any]) -> dict[str, Any]:
    task_name = str(task.get('name', 'dataset_evaluator'))
    mode = str(task.get('mode', 'grid_to_grid'))
    output_dir = ensure_dir(task['output_dir'])
    tables_dir = ensure_dir(output_dir / 'tables')
    metrics_dir = ensure_dir(output_dir / 'metrics')
    figures_dir = ensure_dir(output_dir / 'figures')
    reports_dir = ensure_dir(output_dir / 'reports')

    products = task['products']
    eval_tasks: list[dict[str, Any]] = []

    if mode == 'grid_to_grid':
        reference = task['reference']
        for prod in products:
            name = prod['name']
            eval_tasks.append({
                'name': name,
                'mode': 'grid_to_grid',
                'obs_path': reference['path'],
                'sim_path': prod['path'],
                'obs_var': reference.get('var'),
                'sim_var': prod.get('var'),
                'time_range': (task['time_range']['start'], task['time_range']['end']) if task.get('time_range') else None,
                'time_scale': task.get('time_scale', 'native'),
                'time_agg': task.get('time_agg', 'sum'),
                'overall_metrics': task.get('overall_metrics'),
                'spatial_metrics': task.get('spatial_metrics'),
                'group_by': task.get('group_by'),
                'threshold': task.get('threshold', 1.0),
                'out_table_csv': tables_dir / f'{name}_overall.csv',
                'out_group_csv': tables_dir / f'{name}_group.csv',
                'out_spatial_nc': metrics_dir / f'{name}_spatial.nc',
            })
    elif mode == 'grid_to_station':
        reference = task['reference']
        for prod in products:
            name = prod['name']
            eval_tasks.append({
                'name': name,
                'mode': 'table_to_grid',
                'obs_table_path': reference['path'],
                'obs_table_format': reference.get('format'),
                'sim_nc_path': prod['path'],
                'sim_var': prod.get('var'),
                'time_range': (task['time_range']['start'], task['time_range']['end']) if task.get('time_range') else None,
                'time_scale': task.get('time_scale', 'native'),
                'time_agg': task.get('time_agg', 'sum'),
                'station_col': reference.get('station_col', 'station'),
                'time_col': reference.get('time_col', 'time'),
                'lat_col': reference.get('lat_col', 'lat'),
                'lon_col': reference.get('lon_col', 'lon'),
                'value_col': reference.get('value_col', 'obs'),
                'extract_method': task.get('extract_method', 'nearest'),
                'metrics': task.get('overall_metrics'),
                'group_by': task.get('group_by', 'none'),
                'threshold': task.get('threshold', 1.0),
                'out_station_csv': tables_dir / f'{name}_station.csv',
                'out_group_csv': tables_dir / f'{name}_group.csv',
            })
    else:
        raise ValueError(f'Unsupported mode: {mode}')

    records = run_evaluation_tasks(eval_tasks, report_dir=reports_dir)

    product_names = [p['name'] for p in products]
    if mode == 'grid_to_grid':
        combined_path = tables_dir / 'summary_metrics.csv'
        combined_df = _combine_metric_tables([tables_dir / f'{n}_overall.csv' for n in product_names], product_names, combined_path)
    else:
        combined_path = tables_dir / 'summary_station_metrics.csv'
        combined_df = _combine_metric_tables([tables_dir / f'{n}_station.csv' for n in product_names], product_names, combined_path)

    heatmap_path = ''
    if not combined_df.empty and task.get('make_heatmap', True):
        metric_cols = [c for c in combined_df.columns if c != 'product' and pd.api.types.is_numeric_dtype(combined_df[c])]
        if metric_cols:
            long_df = combined_df.melt(id_vars='product', value_vars=metric_cols, var_name='metric', value_name='value')
            heatmap_path = str(figures_dir / 'metrics_heatmap.png')
            plot_metric_heatmap(
                long_df,
                index_col='product',
                columns_col='metric',
                value_col='value',
                title=task.get('heatmap_title', f'{task_name} metrics heatmap'),
                out_path=heatmap_path,
            )

    summary = {
        'task_name': task_name,
        'mode': mode,
        'output_dir': str(output_dir),
        'combined_metrics_csv': str(combined_path),
        'heatmap_path': heatmap_path,
        'records': [{k: v for k, v in rec.items() if k != '_result'} for rec in records],
    }
    save_json(summary, reports_dir / f'{task_name}_summary.json')
    _write_markdown(summary, reports_dir / f'{task_name}_report.md')
    return summary


def run_dataset_evaluator_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_dataset_evaluator_task(task) for task in tasks if task.get('enabled', True)]
