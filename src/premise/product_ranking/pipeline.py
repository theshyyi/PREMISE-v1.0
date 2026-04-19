from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .fusion import fuse_top_products_mean, fuse_top_products_weighted
from .io import ensure_columns, load_metric_table, save_dataframe
from .methods import average_rank, borda_rank, consensus_rank, simple_metric_ranks, topsis_rank, weighted_sum_rank
from .report import write_excel_book, write_json, write_markdown_summary
from .weights import critic_weights, entropy_weights, equal_weights


def _resolve_metrics(df: pd.DataFrame, product_col: str, metrics: list[str] | None) -> list[str]:
    if metrics is not None:
        ensure_columns(df, [product_col] + metrics)
        return metrics
    candidates = [c for c in df.columns if c != product_col and pd.api.types.is_numeric_dtype(df[c])]
    if not candidates:
        raise ValueError('No numeric metric columns found. Please specify metrics explicitly.')
    return candidates


def _build_weights(metric_df: pd.DataFrame, metric_directions: dict[str, str], weight_method: str) -> pd.Series:
    wm = weight_method.lower()
    if wm == 'equal':
        return equal_weights(list(metric_df.columns))
    if wm == 'critic':
        return critic_weights(metric_df, metric_directions)
    if wm == 'entropy':
        return entropy_weights(metric_df, metric_directions)
    raise ValueError(f'Unsupported weight_method: {weight_method}')


def run_product_ranking_task(task: Mapping[str, Any]) -> dict[str, Any]:
    task_name = str(task['name'])
    input_table = Path(task['input_table'])
    output_dir = Path(task['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    product_col = task.get('product_col', 'product')
    metrics = task.get('metrics')
    metric_directions = dict(task['metric_directions'])
    methods = list(task.get('methods', ['simple_average_rank', 'critic_topsis', 'entropy_weighted_sum', 'consensus']))
    top_n = int(task.get('top_n', 3))
    product_paths = task.get('product_paths')
    fused_output = task.get('fused_output')
    fused_var_name = task.get('fused_var_name')
    fusion_method = str(task.get('fusion_method', 'mean')).lower()
    sheet_name = task.get('sheet_name', 0)

    df = load_metric_table(input_table, sheet_name=sheet_name)
    ensure_columns(df, [product_col])
    metrics = _resolve_metrics(df, product_col, metrics)
    ensure_columns(df, metrics)
    metric_df = df[metrics].apply(pd.to_numeric, errors='coerce')

    results: dict[str, pd.DataFrame] = {}
    weight_results: dict[str, pd.Series] = {}

    if 'simple_metric' in methods:
        results['simple_metric'] = simple_metric_ranks(df[[product_col] + metrics], product_col, metric_directions)

    if 'simple_average_rank' in methods:
        results['simple_average_rank'] = average_rank(df[[product_col] + metrics], product_col, metric_directions)

    if 'borda' in methods:
        results['borda'] = borda_rank(df[[product_col] + metrics], product_col, metric_directions)

    if 'critic_topsis' in methods or 'critic_weighted_sum' in methods:
        w_critic = _build_weights(metric_df, metric_directions, 'critic')
        weight_results['critic'] = w_critic
        if 'critic_topsis' in methods:
            results['critic_topsis'] = topsis_rank(df[[product_col] + metrics], product_col, metric_directions, w_critic)
        if 'critic_weighted_sum' in methods:
            results['critic_weighted_sum'] = weighted_sum_rank(df[[product_col] + metrics], product_col, metric_directions, w_critic)

    if 'entropy_topsis' in methods or 'entropy_weighted_sum' in methods:
        w_entropy = _build_weights(metric_df, metric_directions, 'entropy')
        weight_results['entropy'] = w_entropy
        if 'entropy_topsis' in methods:
            results['entropy_topsis'] = topsis_rank(df[[product_col] + metrics], product_col, metric_directions, w_entropy)
        if 'entropy_weighted_sum' in methods:
            results['entropy_weighted_sum'] = weighted_sum_rank(df[[product_col] + metrics], product_col, metric_directions, w_entropy)

    if 'equal_topsis' in methods or 'equal_weighted_sum' in methods:
        w_equal = _build_weights(metric_df, metric_directions, 'equal')
        weight_results['equal'] = w_equal
        if 'equal_topsis' in methods:
            results['equal_topsis'] = topsis_rank(df[[product_col] + metrics], product_col, metric_directions, w_equal)
        if 'equal_weighted_sum' in methods:
            results['equal_weighted_sum'] = weighted_sum_rank(df[[product_col] + metrics], product_col, metric_directions, w_equal)

    if 'consensus' in methods:
        base = {k: v for k, v in results.items() if 'rank' in v.columns}
        if not base:
            raise ValueError('consensus requested but no prior ranking methods were executed.')
        results['consensus'] = consensus_rank(base, product_col)

    if not results:
        raise ValueError('No ranking methods executed. Check methods configuration.')

    selection_method = str(task.get('selection_method', 'consensus' if 'consensus' in results else list(results.keys())[0]))
    if selection_method not in results:
        raise KeyError(f'selection_method {selection_method} not found in executed results: {list(results)}')
    selected_df = results[selection_method].copy()
    selected_products = selected_df[product_col].head(top_n).tolist()

    selection_out = pd.DataFrame({
        'rank': list(range(1, len(selected_products) + 1)),
        product_col: selected_products,
        'selection_method': selection_method,
    })
    results['selected_top_products'] = selection_out

    fusion_info: dict[str, Any] = {'enabled': False}
    if product_paths and fused_output:
        if fusion_method == 'mean':
            ds_fused = fuse_top_products_mean(product_paths, selected_products, var_name=fused_var_name, output_path=fused_output)
            fusion_info = {
                'enabled': True,
                'fusion_method': 'mean',
                'output_path': str(fused_output),
                'selected_products': selected_products,
                'attrs': dict(ds_fused.attrs),
            }
        elif fusion_method == 'weighted_mean':
            score_col = [c for c in selected_df.columns if c.endswith('_score')]
            if score_col:
                score_col = score_col[0]
                sel = selected_df.set_index(product_col).loc[selected_products, score_col].astype(float)
            else:
                # derive from inverse rank if no score column present
                sel = 1.0 / selected_df.set_index(product_col).loc[selected_products, 'rank'].astype(float)
            weights = (sel / sel.sum()).to_dict()
            ds_fused = fuse_top_products_weighted(product_paths, weights, var_name=fused_var_name, output_path=fused_output)
            fusion_info = {
                'enabled': True,
                'fusion_method': 'weighted_mean',
                'output_path': str(fused_output),
                'selected_products': selected_products,
                'weights': weights,
                'attrs': dict(ds_fused.attrs),
            }
        else:
            raise ValueError(f'Unsupported fusion_method: {fusion_method}')

    excel_path = output_dir / f'{task_name}_ranking_results.xlsx'
    csv_dir = output_dir / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name, dfm in results.items():
        save_dataframe(dfm, csv_dir / f'{name}.csv')
    for wname, wser in weight_results.items():
        save_dataframe(wser.rename('weight').reset_index().rename(columns={'index': 'metric'}), csv_dir / f'weights_{wname}.csv')

    sheets = {k: v for k, v in results.items()}
    for wname, wser in weight_results.items():
        sheets[f'weights_{wname}'] = wser.rename('weight').reset_index().rename(columns={'index': 'metric'})
    write_excel_book(excel_path, sheets)

    summary = {
        'task_name': task_name,
        'input_table': str(input_table),
        'metrics': metrics,
        'metric_directions': metric_directions,
        'methods': methods,
        'selection_method': selection_method,
        'selected_products': selected_products,
        'top_n': top_n,
        'fusion': fusion_info,
    }
    write_json(output_dir / f'{task_name}_summary.json', summary)
    write_markdown_summary(
        output_dir / f'{task_name}_report.md',
        task_name=task_name,
        methods=results,
        selected_products=selected_products,
        fusion_note=(
            f"Fusion enabled: {fusion_info['enabled']}. "
            f"Method={fusion_info.get('fusion_method', '')}. "
            f"Output={fusion_info.get('output_path', '')}"
        ),
    )
    return summary


def run_product_ranking_tasks(tasks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for task in tasks:
        if not bool(task.get('enabled', True)):
            continue
        out.append(run_product_ranking_task(task))
    return out
