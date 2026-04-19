from __future__ import annotations

from typing import Dict, List


def suggest_visualization(result_type: str) -> dict:
    rt = str(result_type).lower()
    mapping = {
        "spatial_nc": {"recommended": ["spatial_map", "multi_spatial_map"], "outputs": ["png"]},
        "metric_table": {"recommended": ["heatmap", "grouped_bar", "boxplot", "violin"], "outputs": ["png"]},
        "timeseries": {"recommended": ["line"], "outputs": ["png"]},
        "grid_compare": {"recommended": ["density_scatter", "taylor", "sal"], "outputs": ["png"]},
        "performance": {"recommended": ["performance_diagram_seasons", "performance_diagram_months", "performance_diagram_regions"], "outputs": ["png"]},
        "ranking": {"recommended": ["ranking_bar", "ranking_heatmap"], "outputs": ["png"]},
    }
    return mapping.get(rt, {"recommended": [], "outputs": ["png"]})
