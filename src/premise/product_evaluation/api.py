
from __future__ import annotations

from .grid import evaluate_grid_pair, evaluate_grid_by_group, compute_spatial_metric_dataset, run_grid_evaluation
from .station import run_station_evaluation
from .pipeline import run_task, run_tasks

__all__ = [
    "evaluate_grid_pair",
    "evaluate_grid_by_group",
    "compute_spatial_metric_dataset",
    "run_grid_evaluation",
    "run_station_evaluation",
    "run_task",
    "run_tasks",
    "describe_ranking_workflow",
]


def describe_ranking_workflow() -> dict:
    """Return a lightweight description of the evaluation-to-ranking workflow.

    This helper is intentionally dependency-light and is used by the quick-start
    documentation to show how product evaluation outputs can feed ranking and
    decision-support workflows. Detailed ranking methods are implemented in
    ``premise.product_ranking``.
    """
    return {
        "steps": [
            "prepare reference and candidate product datasets",
            "harmonize variables, units, grids, and temporal coordinates",
            "compute continuous, event-detection, and spatial diagnostics",
            "summarize metrics by product, region, and time scale",
            "rank products using single-metric, multi-metric, or consensus methods",
        ],
        "ranking_module": "premise.product_ranking",
        "typical_metrics": ["BIAS", "MAE", "RMSE", "CORR", "KGE", "NSE", "POD", "FAR", "CSI"],
    }
