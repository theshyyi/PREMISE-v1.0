from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from premise.basin import process_dataset

BASE_OUT = Path(r"I:\PREMISE-v1.0\basin_output")
REPORT_DIR = BASE_OUT / "_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TASKS: list[dict[str, Any]] = [
    {
        "name": "bbox_clip_demo",
        "enabled": True,
        "input": r"I:\PREMISE-v1.0\conversion_output\imerg_hdf5_demo.nc",
        "output": str(BASE_OUT / "imerg_bbox_clip.nc"),
        "bbox": {"min_lon": 100.0, "max_lon": 120.0, "min_lat": 20.0, "max_lat": 35.0},
        "time_range": {"start": "2020-02-01", "end": "2020-02-01"},
        "rename_map": {"pr": "precip"},
        "paper_note": "Bounding-box clipping with variable rename.",
    },
    {
        "name": "vector_clip_demo",
        "enabled": True,
        "input": r"I:\PREMISE-v1.0\conversion_output\imerg_hdf5_demo.nc",
        "output": str(BASE_OUT / "imerg_vector_clip.nc"),
        "vector_path": r"I:\博士论文2027\博士中期考核\Chinese_Climate\Chinese_climate.shp",
        "region_field": None,
        "region_values": None,
        "time_range": {"start": "2020-02-01", "end": "2020-02-01"},
        "rename_map": {"pr": "precip"},
        "paper_note": "Vector-based clipping using shapefile/GeoJSON.",
    },
    {
        "name": "vector_clip_resample_demo",
        "enabled": False,
        "input": r"I:\PREMISE-v1.0\conversion_output\era5_grib_demo.nc",
        "output": str(BASE_OUT / "era5_vector_resampled.nc"),
        "vector_path": r"I:\YOUR_VECTOR\basin.geojson",
        "time_range": None,
        "rename_map": {"t2m": "tas"},
        "target_resolution": 0.25,
        "resample_method": "linear",
        "paper_note": "Vector clipping with 0.25 degree resampling.",
    },
]


def _write_records(records: list[dict[str, Any]]) -> None:
    csv_path = REPORT_DIR / "basin_summary.csv"
    json_path = REPORT_DIR / "basin_summary.json"
    md_path = REPORT_DIR / "basin_report.md"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name", "status", "input", "output", "elapsed_seconds",
                "paper_note", "dims_before", "dims_after", "error_type", "error_message"
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "name": r.get("name"),
                "status": r.get("status"),
                "input": r.get("input"),
                "output": r.get("output"),
                "elapsed_seconds": r.get("elapsed_seconds"),
                "paper_note": r.get("paper_note", ""),
                "dims_before": json.dumps((r.get("before") or {}).get("dims", {}), ensure_ascii=False),
                "dims_after": json.dumps((r.get("after") or {}).get("dims", {}), ensure_ascii=False),
                "error_type": r.get("error_type", ""),
                "error_message": r.get("error_message", ""),
            })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    lines = ["# Basin Processing Report", ""]
    for r in records:
        lines.append(f"## {r['name']}")
        lines.append(f"- status: {r['status']}")
        lines.append(f"- input: `{r['input']}`")
        lines.append(f"- output: `{r['output']}`")
        if r['status'] == 'SUCCESS':
            lines.append(f"- elapsed_seconds: {r['elapsed_seconds']}")
            lines.append(f"- dims_before: `{json.dumps((r.get('before') or {}).get('dims', {}), ensure_ascii=False)}`")
            lines.append(f"- dims_after: `{json.dumps((r.get('after') or {}).get('dims', {}), ensure_ascii=False)}`")
        else:
            lines.append(f"- error: {r.get('error_type')}: {r.get('error_message')}")
        if r.get('paper_note'):
            lines.append(f"- note: {r['paper_note']}")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def main() -> None:
    records: list[dict[str, Any]] = []
    for task in TASKS:
        if not task.get("enabled", True):
            print(f"[SKIP] {task['name']}")
            continue
        print("=" * 90)
        print(f"[RUN ] {task['name']}")
        try:
            report = process_dataset(
                task['input'],
                task['output'],
                bbox=task.get('bbox'),
                vector_path=task.get('vector_path'),
                region_field=task.get('region_field'),
                region_values=task.get('region_values'),
                time_range=task.get('time_range'),
                rename_map=task.get('rename_map'),
                target_resolution=task.get('target_resolution'),
                target_grid_path=task.get('target_grid_path'),
                resample_method=task.get('resample_method', 'linear'),
                report_json=REPORT_DIR / f"{task['name']}.json",
            )
            report['name'] = task['name']
            report['status'] = 'SUCCESS'
            report['paper_note'] = task.get('paper_note', '')
            records.append(report)
            print(f"[DONE] {task['name']} -> {task['output']}")
            print(f"[TIME] {report['elapsed_seconds']} s")
        except Exception as e:
            records.append({
                'name': task['name'],
                'status': 'FAILED',
                'input': task['input'],
                'output': task['output'],
                'elapsed_seconds': None,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'paper_note': task.get('paper_note', ''),
            })
            print(f"[FAIL] {task['name']}: {type(e).__name__}: {e}")

    _write_records(records)
    print(f"[REPORT] {REPORT_DIR}")


if __name__ == '__main__':
    main()
