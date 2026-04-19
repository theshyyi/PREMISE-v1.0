from __future__ import annotations

from premise.product_ranking import run_product_ranking_tasks


TASKS = [
    {
        'name': 'precip_product_ranking_demo',
        'enabled': False,
        'input_table': r'I:\PREMISE-v1.0\product_evaluation_output\precip_metrics_summary.csv',
        'output_dir': r'I:\PREMISE-v1.0\product_ranking_output\precip_product_ranking_demo',
        'product_col': 'product',
        'metrics': ['BIAS', 'MAE', 'RMSE', 'CORR', 'KGE', 'NSE', 'POD', 'FAR', 'CSI', 'FBIAS', 'HSS'],
        'metric_directions': {
            'BIAS': 'cost',
            'MAE': 'cost',
            'RMSE': 'cost',
            'CORR': 'benefit',
            'KGE': 'benefit',
            'NSE': 'benefit',
            'POD': 'benefit',
            'FAR': 'cost',
            'CSI': 'benefit',
            'FBIAS': 'cost',
            'HSS': 'benefit',
        },
        'methods': [
            'simple_metric',
            'simple_average_rank',
            'borda',
            'critic_topsis',
            'entropy_weighted_sum',
            'consensus',
        ],
        'selection_method': 'consensus',
        'top_n': 3,
        'fusion_method': 'mean',
        'fused_var_name': 'pr',
        'product_paths': {
            'CHIRPS': r'I:\PREMISE-v1.0\converted_products\CHIRPS.nc',
            'IMERG': r'I:\PREMISE-v1.0\converted_products\IMERG.nc',
            'MSWEP': r'I:\PREMISE-v1.0\converted_products\MSWEP.nc',
        },
        'fused_output': r'I:\PREMISE-v1.0\product_ranking_output\precip_product_ranking_demo\top3_fused_mean.nc',
    },
    {
        'name': 'temperature_product_ranking_demo',
        'enabled': False,
        'input_table': r'I:\PREMISE-v1.0\product_evaluation_output\temperature_metrics_summary.xlsx',
        'sheet_name': 0,
        'output_dir': r'I:\PREMISE-v1.0\product_ranking_output\temperature_product_ranking_demo',
        'product_col': 'product',
        'metrics': ['MAE', 'RMSE', 'CORR', 'KGE', 'NSE'],
        'metric_directions': {
            'MAE': 'cost',
            'RMSE': 'cost',
            'CORR': 'benefit',
            'KGE': 'benefit',
            'NSE': 'benefit',
        },
        'methods': ['simple_average_rank', 'critic_topsis', 'equal_weighted_sum', 'consensus'],
        'selection_method': 'critic_topsis',
        'top_n': 2,
    },
]


if __name__ == '__main__':
    summaries = run_product_ranking_tasks(TASKS)
    for s in summaries:
        print(f"[DONE] {s['task_name']} -> selected: {s['selected_products']}")
