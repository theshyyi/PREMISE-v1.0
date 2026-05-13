from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import xarray as xr

from .drought import calc_spi, calc_spei, calc_sri, calc_sti
from .extremes import rx1day, rx5day, prcptot, sdii, r10mm, r20mm, r95p, r99p, cdd, cwd
from .io import open_dataset, save_dataset, subset_time
from .report import write_csv, write_json, write_markdown


EXTREME_FUNCS = {
    "rx1day": rx1day,
    "rx5day": rx5day,
    "prcptot": prcptot,
    "sdii": sdii,
    "r10mm": r10mm,
    "r20mm": r20mm,
    "r95p": r95p,
    "r99p": r99p,
    "cdd": cdd,
    "cwd": cwd,
}


HYDRO_FUNCS = {
    "spi": calc_spi,
    "spei": calc_spei,
    "sri": calc_sri,
    "sti": calc_sti,
}


def compute_extreme_indices(ds: xr.Dataset, config: dict[str, Any]) -> xr.Dataset:
    var_name = config.get("var_name", "pr")
    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")
    da = ds[var_name]
    requested = config.get("indices") or list(EXTREME_FUNCS.keys())
    params = config.get("params", {})

    out = xr.Dataset()
    for name in requested:
        if name not in EXTREME_FUNCS:
            raise KeyError(f"Unsupported extreme index: {name}")
        kw = dict(params.get(name, {}))
        out[name] = EXTREME_FUNCS[name](da, **kw)
    return out


def compute_hydroclimatic_indices(ds: xr.Dataset, config: dict[str, Any]) -> xr.Dataset:
    requested = config.get("indices") or ["spi"]
    vars_cfg = config.get("variables", {})
    params = config.get("params", {})
    out = xr.Dataset()

    for name in requested:
        if name == "spi":
            var = vars_cfg.get("precip", "pr")
            out[f"SPI_{params.get(name, {}).get('scale', 3)}"] = calc_spi(ds[var], **dict(params.get(name, {})))
        elif name == "spei":
            pvar = vars_cfg.get("precip", "pr")
            petvar = vars_cfg.get("pet", "pet")
            scale = params.get(name, {}).get("scale", 3)
            out[f"SPEI_{scale}"] = calc_spei(ds[pvar], ds[petvar], **dict(params.get(name, {})))
        elif name == "sri":
            qvar = vars_cfg.get("runoff", "qtot")
            scale = params.get(name, {}).get("scale", 3)
            out[f"SRI_{scale}"] = calc_sri(ds[qvar], **dict(params.get(name, {})))
        elif name == "sti":
            tvar = vars_cfg.get("temp", "tas")
            scale = params.get(name, {}).get("scale", 1)
            out[f"STI_{scale}"] = calc_sti(ds[tvar], **dict(params.get(name, {})))
        else:
            raise KeyError(f"Unsupported hydroclimatic index: {name}")
    return out


def _summarize_output(ds: xr.Dataset) -> tuple[dict[str, int], list[str]]:
    dims = {k: int(v) for k, v in ds.sizes.items()}
    vars_ = list(ds.data_vars)
    return dims, vars_


def run_climate_indices_task(task: dict[str, Any]) -> dict[str, Any]:
    name = task.get("name", "task")
    family = task.get("family", "extremes")
    input_path = Path(task["input"])
    output_path = Path(task["output"])

    record: dict[str, Any] = {
        "name": name,
        "family": family,
        "input": str(input_path),
        "output": str(output_path),
        "time_range": task.get("time_range"),
        "note": task.get("note", ""),
        "status": "FAILED",
        "start_time": None,
        "end_time": None,
        "elapsed_seconds": None,
        "variables": None,
        "dims": None,
        "error_type": "",
        "error_message": "",
    }

    start = time.perf_counter()
    record["start_time"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    try:
        ds = open_dataset(input_path, chunks=task.get("chunks"))
        ds = subset_time(ds, task.get("time_range"))

        if family == "extremes":
            out = compute_extreme_indices(ds, task.get("config", {}))
        elif family == "hydroclimatic":
            out = compute_hydroclimatic_indices(ds, task.get("config", {}))
        else:
            raise ValueError("family must be 'extremes' or 'hydroclimatic'")

        save_dataset(out, output_path, comp_level=int(task.get("comp_level", 4)))
        dims, vars_ = _summarize_output(out)
        record["variables"] = vars_
        record["dims"] = dims
        record["status"] = "SUCCESS"
        ds.close()
    except Exception as e:
        record["error_type"] = type(e).__name__
        record["error_message"] = str(e)
    record["end_time"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    record["elapsed_seconds"] = round(time.perf_counter() - start, 4)
    return record


def run_climate_indices_tasks(tasks: list[dict[str, Any]], report_dir: str | Path | None = None) -> list[dict[str, Any]]:
    records = [run_climate_indices_task(task) for task in tasks if task.get("enabled", True)]
    if report_dir is not None:
        report_dir = Path(report_dir)
        write_csv(records, report_dir / "climate_indices_summary.csv")
        write_json(records, report_dir / "climate_indices_summary.json")
        write_markdown(records, report_dir / "climate_indices_report.md")
    return records
