
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .grid import run_grid_evaluation
from .report import write_summary_csv, write_summary_json, write_summary_md
from .station import run_station_evaluation


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    mode = task["mode"]
    name = task.get("name", mode)
    out_record = {
        "name": name,
        "mode": mode,
        "status": "FAILED",
        "time_scale": task.get("time_scale", "native"),
        "group_by": task.get("group_by", "none"),
        "elapsed_seconds": None,
        "output_main": "",
        "error_message": "",
    }

    t0 = time.perf_counter()
    try:
        if mode == "grid_to_grid":
            res = run_grid_evaluation(
                obs_path=task["obs_path"],
                sim_path=task["sim_path"],
                obs_var=task.get("obs_var"),
                sim_var=task.get("sim_var"),
                time_range=task.get("time_range"),
                time_scale=task.get("time_scale", "native"),
                time_agg=task.get("time_agg", "sum"),
                overall_metrics=task.get("overall_metrics"),
                spatial_metrics=task.get("spatial_metrics"),
                group_by=task.get("group_by"),
                threshold=task.get("threshold", 1.0),
                out_table_csv=task.get("out_table_csv"),
                out_group_csv=task.get("out_group_csv"),
                out_spatial_nc=task.get("out_spatial_nc"),
            )
            out_record["output_main"] = str(task.get("out_table_csv") or task.get("out_spatial_nc") or "")
        elif mode == "table_to_grid":
            res = run_station_evaluation(
                obs_table_path=task["obs_table_path"],
                sim_nc_path=task["sim_nc_path"],
                obs_table_format=task.get("obs_table_format"),
                sim_var=task.get("sim_var"),
                time_range=task.get("time_range"),
                time_scale=task.get("time_scale", "native"),
                time_agg=task.get("time_agg", "sum"),
                station_col=task.get("station_col", "station"),
                time_col=task.get("time_col", "time"),
                lat_col=task.get("lat_col", "lat"),
                lon_col=task.get("lon_col", "lon"),
                value_col=task.get("value_col", "obs"),
                extract_method=task.get("extract_method", "nearest"),
                metrics=task.get("metrics"),
                group_by=task.get("group_by", "none"),
                threshold=task.get("threshold", 1.0),
                out_station_csv=task.get("out_station_csv"),
                out_group_csv=task.get("out_group_csv"),
            )
            out_record["output_main"] = str(task.get("out_station_csv") or "")
        else:
            raise ValueError(f"不支持的 mode: {mode}")

        out_record["status"] = "SUCCESS"
    except Exception as e:
        out_record["error_message"] = f"{type(e).__name__}: {e}"
        res = None
    finally:
        out_record["elapsed_seconds"] = round(time.perf_counter() - t0, 4)
    out_record["_result"] = res
    return out_record


def run_tasks(tasks: list[dict[str, Any]], *, report_dir: str | Path | None = None) -> list[dict[str, Any]]:
    records = []
    for task in tasks:
        if not task.get("enabled", True):
            continue
        records.append(run_task(task))

    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        clean_records = [{k: v for k, v in r.items() if k != "_result"} for r in records]
        write_summary_csv(clean_records, report_dir / "product_evaluation_summary.csv")
        write_summary_json(clean_records, report_dir / "product_evaluation_summary.json")
        write_summary_md(clean_records, report_dir / "product_evaluation_report.md")
    return records
