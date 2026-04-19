from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from premise.product_ranking.pipeline import run_product_ranking_task
from premise.visualization.ranking import plot_ranking_bar, plot_ranking_score_heatmap

from .common import ensure_dir, save_json


def run_product_ranker_task(task: dict[str, Any]) -> dict[str, Any]:
    task_name = str(task.get('name', 'product_ranker'))
    output_dir = ensure_dir(task['output_dir'])
    figures_dir = ensure_dir(output_dir / 'figures')

    summary = run_product_ranking_task(task)

    selection_method = summary['selection_method']
    csv_dir = Path(output_dir) / 'csv'
    selected_csv = csv_dir / f'{selection_method}.csv'
    ranking_bar_path = ''
    ranking_heatmap_path = ''

    if selected_csv.exists() and task.get('make_ranking_bar', True):
        df = pd.read_csv(selected_csv)
        product_col = task.get('product_col', 'product')
        score_col = next((c for c in df.columns if c.endswith('_score')), None)
        if score_col is not None:
            ranking_bar_path = str(figures_dir / 'ranking_bar.png')
            plot_ranking_bar(
                df,
                product_col=product_col,
                score_col=score_col,
                top_n=task.get('plot_top_n'),
                title=task.get('ranking_bar_title', f'{task_name} ranking bar'),
                out_path=ranking_bar_path,
            )

    if task.get('make_score_heatmap', True):
        frames = []
        product_col = task.get('product_col', 'product')
        for csv_path in csv_dir.glob('*.csv'):
            method = csv_path.stem
            if method.startswith('weights_') or method == 'selected_top_products':
                continue
            dfm = pd.read_csv(csv_path)
            score_col = next((c for c in dfm.columns if c.endswith('_score')), None)
            if score_col is None:
                continue
            frames.append(dfm[[product_col, score_col]].rename(columns={score_col: 'score'}).assign(method=method))
        if frames:
            long_df = pd.concat(frames, ignore_index=True)
            ranking_heatmap_path = str(figures_dir / 'ranking_score_heatmap.png')
            plot_ranking_score_heatmap(
                long_df,
                product_col=product_col,
                method_col='method',
                score_col='score',
                title=task.get('ranking_heatmap_title', f'{task_name} ranking score heatmap'),
                out_path=ranking_heatmap_path,
            )

    app_summary = {
        'task_name': task_name,
        'output_dir': str(output_dir),
        'selection_method': summary['selection_method'],
        'selected_products': summary['selected_products'],
        'ranking_bar_path': ranking_bar_path,
        'ranking_heatmap_path': ranking_heatmap_path,
        'summary': summary,
    }
    save_json(app_summary, Path(output_dir) / f'{task_name}_app_summary.json')
    return app_summary


def run_product_ranker_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_product_ranker_task(task) for task in tasks if task.get('enabled', True)]
