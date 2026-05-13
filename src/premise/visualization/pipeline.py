from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import xarray as xr

from .styles import setup_mpl_fonts
from .maps import plot_spatial_field, plot_multi_spatial_fields
from .timeseries import plot_timeseries_lines
from .distributions import plot_metric_heatmap, plot_grouped_bar, plot_boxplot_groups, plot_violin_groups
from .diagnostics import (
    scatter_density_product,
    multi_scatter_density_products,
    time_group_scatter_density_product,
    plot_taylor_diagram,
    plot_sal_scatter,
    plot_performance_diagram_seasons,
    plot_performance_diagram_months,
    plot_performance_diagram_regions,
)
from .ranking import plot_ranking_bar, plot_ranking_score_heatmap
from .auto import suggest_visualization


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def run_visualization_task(task: Dict[str, Any]) -> Dict[str, Any]:
    setup_mpl_fonts(use_chinese=bool(task.get("use_chinese", True)), base_size=int(task.get("base_size", 10)))
    plot_type = str(task["plot_type"]).lower()
    out_path = task.get("out_path")

    if plot_type == "spatial_map":
        ds = xr.open_dataset(task["input"])
        try:
            da = ds[task["var_name"]]
            fig, ax = plot_spatial_field(da, out_path=out_path, **task.get("kwargs", {}))
        finally:
            ds.close()
        return {"status": "SUCCESS", "plot_type": plot_type, "out_path": str(out_path)}

    if plot_type == "multi_spatial_map":
        ds = xr.open_dataset(task["input"])
        try:
            fig, ax = plot_multi_spatial_fields(ds, variables=task.get("variables"), out_path=out_path, **task.get("kwargs", {}))
        finally:
            ds.close()
        return {"status": "SUCCESS", "plot_type": plot_type, "out_path": str(out_path)}

    if plot_type == "timeseries":
        src = task["input"]
        series_dict = {}
        if isinstance(src, dict):
            for name, p in src.items():
                if str(p).endswith(".nc"):
                    ds = xr.open_dataset(p)
                    try:
                        series_dict[name] = ds[task["var_name"]]
                    finally:
                        ds.close()
                else:
                    df = _read_table(p)
                    series_dict[name] = df
        fig, ax = plot_timeseries_lines(series_dict, out_path=out_path, **task.get("kwargs", {}))
        return {"status": "SUCCESS", "plot_type": plot_type, "out_path": str(out_path)}

    if plot_type in {"heatmap", "grouped_bar", "boxplot", "violin", "ranking_bar", "ranking_heatmap"}:
        df = _read_table(task["input"])
        kwargs = task.get("kwargs", {})
        if plot_type == "heatmap":
            plot_metric_heatmap(df, out_path=out_path, **kwargs)
        elif plot_type == "grouped_bar":
            plot_grouped_bar(df, out_path=out_path, **kwargs)
        elif plot_type == "boxplot":
            data_dict = {str(k): g[kwargs["value_col"]].dropna().values for k, g in df.groupby(kwargs["group_col"])}
            kw = {k: v for k, v in kwargs.items() if k not in {"group_col", "value_col"}}
            plot_boxplot_groups(data_dict, out_path=out_path, **kw)
        elif plot_type == "violin":
            data_dict = {str(k): g[kwargs["value_col"]].dropna().values for k, g in df.groupby(kwargs["group_col"])}
            kw = {k: v for k, v in kwargs.items() if k not in {"group_col", "value_col"}}
            plot_violin_groups(data_dict, out_path=out_path, **kw)
        elif plot_type == "ranking_bar":
            plot_ranking_bar(df, out_path=out_path, **kwargs)
        elif plot_type == "ranking_heatmap":
            plot_ranking_score_heatmap(df, out_path=out_path, **kwargs)
        return {"status": "SUCCESS", "plot_type": plot_type, "out_path": str(out_path)}

    if plot_type == "density_scatter":
        ref = xr.open_dataset(task["ref_input"])
        sim = xr.open_dataset(task["sim_input"])
        try:
            scatter_density_product(ref[task["ref_var"]], sim[task["sim_var"]], out_path=out_path, **task.get("kwargs", {}))
        finally:
            ref.close(); sim.close()
        return {"status": "SUCCESS", "plot_type": plot_type, "out_path": str(out_path)}

    if plot_type == "suggest":
        return {"status": "SUCCESS", "plot_type": plot_type, "suggestion": suggest_visualization(task["result_type"])}

    raise ValueError(f"Unsupported plot_type: {plot_type}")


def run_visualization_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [run_visualization_task(t) for t in tasks]
