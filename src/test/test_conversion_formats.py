from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import xarray as xr

from premise.conversion import (
    convert_binary_to_nc,
    convert_geotiff_to_monthly_nc,
    convert_grib_to_nc,
    convert_hdf_to_nc,
)

# =========================================================
# 1. 基本配置
# =========================================================

BASE_OUTPUT_DIR = Path(r"I:\PREMISE-v1.0\conversion_output")
REPORT_DIR = BASE_OUTPUT_DIR / "_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = REPORT_DIR / "conversion_run.log"
CSV_PATH = REPORT_DIR / "conversion_summary.csv"
JSON_PATH = REPORT_DIR / "conversion_summary.json"
MD_PATH = REPORT_DIR / "conversion_report.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# =========================================================
# 2. 真实数据任务
# =========================================================

TASKS: list[dict[str, Any]] = [
    {
        "name": "imerg_hdf5_demo",
        "kind": "hdf",
        "input": r"I:\PREMISE-v1.0\tests_output\imerg_hdf5_smoke\3B-HHR.MS.MRG.3IMERG.20200201-S000000-E002959.0000.V07B.HDF5",
        "output": str(BASE_OUTPUT_DIR / "imerg_hdf5_demo.nc"),
        "enabled": True,
        "kwargs": {
            "group": "Grid",
            "dataset": "precipitation",
            "var_name": "pr",
            "units": "mm/hr",
            "lat_path": "lat",
            "lon_path": "lon",
        },
        "paper_note": "IMERG HDF5 precipitation product converted to NetCDF for unified downstream analysis."
    },
    {
        "name": "era5_grib_demo",
        "kind": "grib",
        "input": r"I:\PREMISE-v1.0\tests_output\era5land_hourly_smoke\era5land_hourly_2m_temperature_20200101_20200101.grib",
        "output": str(BASE_OUTPUT_DIR / "era5_grib_demo.nc"),
        "enabled": True,
        "kwargs": {
            "var": "t2m",
        },
        "paper_note": "ERA5-Land hourly 2 m temperature GRIB file converted to NetCDF."
    },
    # 你可以继续往下加 binary / geotiff 等
]

# =========================================================
# 3. 工具函数
# =========================================================

def _file_size_mb(path: Path) -> float | None:
    if not path.exists() or not path.is_file():
        return None
    return round(path.stat().st_size / 1024 / 1024, 4)


def _safe_open_dataset(nc_path: Path) -> xr.Dataset:
    engines = ["h5netcdf", "netcdf4", "scipy", None]
    last_error = None

    for engine in engines:
        try:
            if engine is None:
                return xr.open_dataset(nc_path)
            return xr.open_dataset(nc_path, engine=engine)
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"无法打开输出 nc 文件: {nc_path}\n最后错误: {last_error}")


def _extract_nc_summary(nc_path: Path) -> dict[str, Any]:
    with _safe_open_dataset(nc_path) as ds:
        dims_dict = {k: int(v) for k, v in ds.sizes.items()}
        data_vars = list(ds.data_vars)
        coords = list(ds.coords)

        var_summary = {}
        for var in data_vars:
            da = ds[var]
            var_summary[var] = {
                "shape": [int(x) for x in da.shape],
                "dtype": str(da.dtype),
                "units": da.attrs.get("units", ""),
                "dims": list(da.dims),
                "missing_count": int(da.isnull().sum().item()) if da.ndim <= 3 else None,
            }

        return {
            "dims": dims_dict,
            "coords": coords,
            "data_vars": data_vars,
            "var_summary": var_summary,
        }


def _load_binary_meta(task: dict[str, Any]) -> dict[str, Any]:
    if "meta" in task and task["meta"] is not None:
        return task["meta"]

    meta_json = task.get("meta_json")
    if meta_json:
        meta_path = Path(meta_json)
        if not meta_path.exists():
            raise FileNotFoundError(f"binary meta_json 不存在: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError("binary 类型任务必须提供 meta 或 meta_json。")


def _run_binary(task: dict[str, Any]) -> None:
    convert_binary_to_nc(
        Path(task["input"]),
        Path(task["output"]),
        meta=_load_binary_meta(task),
    )


def _run_hdf(task: dict[str, Any]) -> None:
    convert_hdf_to_nc(
        Path(task["input"]),
        Path(task["output"]),
        **task.get("kwargs", {}),
    )


def _run_geotiff_monthly(task: dict[str, Any]) -> None:
    convert_geotiff_to_monthly_nc(
        Path(task["input"]),
        Path(task["output"]),
        **task.get("kwargs", {}),
    )


def _run_grib(task: dict[str, Any]) -> None:
    convert_grib_to_nc(
        Path(task["input"]),
        Path(task["output"]),
        **task.get("kwargs", {}),
    )


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    name = task["name"]
    kind = str(task["kind"]).lower()
    enabled = bool(task.get("enabled", True))

    record: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "enabled": enabled,
        "input": str(task["input"]),
        "output": str(task["output"]),
        "paper_note": task.get("paper_note", ""),
        "status": "SKIPPED",
        "start_time": None,
        "end_time": None,
        "elapsed_seconds": None,
        "input_size_mb": None,
        "output_size_mb": None,
        "dims": None,
        "data_vars": None,
        "var_summary": None,
        "error_type": "",
        "error_message": "",
    }

    if not enabled:
        logging.info(f"[SKIP] {name} 未启用")
        return record

    input_path = Path(task["input"])
    output_path = Path(task["output"])

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.is_file():
            logging.info(f"[INFO] 删除旧文件: {output_path}")
            output_path.unlink()
        else:
            raise IsADirectoryError(f"输出路径已存在但不是文件: {output_path}")

    record["input_size_mb"] = _file_size_mb(input_path)
    start_dt = datetime.now()
    start_perf = time.perf_counter()

    record["start_time"] = start_dt.strftime("%Y-%m-%d %H:%M:%S")

    logging.info("=" * 90)
    logging.info(f"[RUN ] {name}")
    logging.info(f"       kind   : {kind}")
    logging.info(f"       input  : {input_path}")
    logging.info(f"       output : {output_path}")

    try:
        if kind == "binary":
            _run_binary(task)
        elif kind == "hdf":
            _run_hdf(task)
        elif kind == "geotiff_monthly":
            _run_geotiff_monthly(task)
        elif kind == "grib":
            _run_grib(task)
        else:
            raise ValueError(f"不支持的 kind: {kind}")

        end_dt = datetime.now()
        elapsed = time.perf_counter() - start_perf

        record["end_time"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        record["elapsed_seconds"] = round(elapsed, 4)
        record["output_size_mb"] = _file_size_mb(output_path)

        summary = _extract_nc_summary(output_path)
        record["dims"] = summary["dims"]
        record["data_vars"] = summary["data_vars"]
        record["var_summary"] = summary["var_summary"]
        record["status"] = "SUCCESS"

        logging.info(f"[DONE] 转换完成: {output_path}")
        logging.info(f"[TIME] 用时: {record['elapsed_seconds']} s")
        logging.info(f"[SIZE] 输入: {record['input_size_mb']} MB | 输出: {record['output_size_mb']} MB")
        logging.info(f"[VARS] {record['data_vars']}")
        logging.info(f"[DIMS] {record['dims']}")

    except Exception as e:
        end_dt = datetime.now()
        elapsed = time.perf_counter() - start_perf
        record["end_time"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        record["elapsed_seconds"] = round(elapsed, 4)
        record["status"] = "FAILED"
        record["error_type"] = type(e).__name__
        record["error_message"] = str(e)

        logging.error(f"[FAIL] {name} 转换失败")
        logging.error(f"       {type(e).__name__}: {e}")

    return record


def _write_csv(records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name", "kind", "enabled", "status",
        "input", "output",
        "start_time", "end_time", "elapsed_seconds",
        "input_size_mb", "output_size_mb",
        "data_vars", "dims",
        "paper_note",
        "error_type", "error_message",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "name": r["name"],
                "kind": r["kind"],
                "enabled": r["enabled"],
                "status": r["status"],
                "input": r["input"],
                "output": r["output"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "elapsed_seconds": r["elapsed_seconds"],
                "input_size_mb": r["input_size_mb"],
                "output_size_mb": r["output_size_mb"],
                "data_vars": json.dumps(r["data_vars"], ensure_ascii=False) if r["data_vars"] is not None else "",
                "dims": json.dumps(r["dims"], ensure_ascii=False) if r["dims"] is not None else "",
                "paper_note": r["paper_note"],
                "error_type": r["error_type"],
                "error_message": r["error_message"],
            })


def _write_json(records: list[dict[str, Any]]) -> None:
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _write_markdown(records: list[dict[str, Any]]) -> None:
    success_records = [r for r in records if r["status"] == "SUCCESS"]
    failed_records = [r for r in records if r["status"] == "FAILED"]

    lines = []
    lines.append("# Conversion Report\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 1. Overall Summary\n")
    lines.append(f"- Total tasks: {len(records)}")
    lines.append(f"- Successful tasks: {len(success_records)}")
    lines.append(f"- Failed tasks: {len(failed_records)}\n")

    lines.append("## 2. Successful Conversions\n")
    lines.append("| Name | Kind | Variables | Dimensions | Time (s) | Input Size (MB) | Output Size (MB) | Note |")
    lines.append("|---|---|---|---|---:|---:|---:|---|")
    for r in success_records:
        lines.append(
            f"| {r['name']} | {r['kind']} | "
            f"{', '.join(r['data_vars'] or [])} | "
            f"`{json.dumps(r['dims'], ensure_ascii=False)}` | "
            f"{r['elapsed_seconds']} | {r['input_size_mb']} | {r['output_size_mb']} | "
            f"{r['paper_note']} |"
        )
    lines.append("")

    lines.append("## 3. Variable Details\n")
    for r in success_records:
        lines.append(f"### {r['name']}\n")
        lines.append(f"- Input: `{r['input']}`")
        lines.append(f"- Output: `{r['output']}`")
        lines.append(f"- Conversion time: `{r['elapsed_seconds']} s`")
        lines.append(f"- Dimensions: `{json.dumps(r['dims'], ensure_ascii=False)}`")
        if r["var_summary"]:
            for var, info in r["var_summary"].items():
                lines.append(
                    f"- Variable `{var}`: shape={info['shape']}, dtype={info['dtype']}, "
                    f"units={info['units']}, dims={info['dims']}, missing_count={info['missing_count']}"
                )
        if r["paper_note"]:
            lines.append(f"- Manuscript note: {r['paper_note']}")
        lines.append("")

    if failed_records:
        lines.append("## 4. Failed Conversions\n")
        lines.append("| Name | Kind | Error Type | Error Message |")
        lines.append("|---|---|---|---|")
        for r in failed_records:
            lines.append(
                f"| {r['name']} | {r['kind']} | {r['error_type']} | {r['error_message']} |"
            )
        lines.append("")

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    logging.info("开始执行真实数据 conversion ...")
    records: list[dict[str, Any]] = []

    for task in TASKS:
        record = _run_one(task)
        records.append(record)

    _write_csv(records)
    _write_json(records)
    _write_markdown(records)

    logging.info("=" * 90)
    logging.info("全部任务执行完成")
    logging.info(f"CSV 记录文件 : {CSV_PATH}")
    logging.info(f"JSON记录文件 : {JSON_PATH}")
    logging.info(f"MD   报告文件 : {MD_PATH}")
    logging.info(f"LOG  日志文件 : {LOG_PATH}")


if __name__ == "__main__":
    main()