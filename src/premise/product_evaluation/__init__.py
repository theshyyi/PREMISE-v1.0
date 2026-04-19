"""Unified product evaluation module.

Scope
-----
This module is only responsible for evaluation/comparison workflows:
- grid_to_grid: reference NetCDF vs product NetCDF
- table_to_grid: station/observation table vs product NetCDF
- time subsetting and temporal aggregation
- overall / grouped metrics tables
- spatial metric fields exported as NetCDF

Climate and extreme indices are handled in the separate `premise.climate_indices`
module and are intentionally not exposed here.
"""
from .api import (
    evaluate_grid_pair,
    evaluate_grid_by_group,
    compute_spatial_metric_dataset,
    run_grid_evaluation,
    run_station_evaluation,
    run_task,
    run_tasks,
)
from .metrics import *

__all__ = [
    "evaluate_grid_pair",
    "evaluate_grid_by_group",
    "compute_spatial_metric_dataset",
    "run_grid_evaluation",
    "run_station_evaluation",
    "run_task",
    "run_tasks",
]
