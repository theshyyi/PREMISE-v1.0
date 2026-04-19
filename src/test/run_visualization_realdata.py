from __future__ import annotations

from premise.visualization import run_visualization_tasks

TASKS = [
    {
        "plot_type": "suggest",
        "result_type": "metric_table",
        "use_chinese": True,
    },
    {
        "plot_type": "spatial_map",
        "input": r"I:\PREMISE-v1.0\product_evaluation_output\spatial_metrics.nc",
        "var_name": "RMSE",
        "out_path": r"I:\PREMISE-v1.0\visualization_output\spatial_rmse.png",
        "use_chinese": True,
        "kwargs": {
            "title": "Spatial RMSE",
            "cbar_label": "RMSE",
            "cmap": "viridis",
        },
    },
    {
        "plot_type": "heatmap",
        "input": r"I:\PREMISE-v1.0\product_evaluation_output\overall_metrics.csv",
        "out_path": r"I:\PREMISE-v1.0\visualization_output\metrics_heatmap.png",
        "use_chinese": True,
        "kwargs": {
            "index_col": "product",
            "columns_col": "metric",
            "value_col": "value",
            "title": "Metric heatmap",
            "cmap": "RdYlBu_r",
        },
    },
    {
        "plot_type": "ranking_bar",
        "input": r"I:\PREMISE-v1.0\product_ranking_output\ranking_scores.csv",
        "out_path": r"I:\PREMISE-v1.0\visualization_output\ranking_bar.png",
        "use_chinese": True,
        "kwargs": {
            "product_col": "product",
            "score_col": "score",
            "top_n": 10,
            "ascending": False,
            "title": "Top-ranked products",
        },
    },
]

if __name__ == "__main__":
    results = run_visualization_tasks(TASKS)
    for r in results:
        print(r)
