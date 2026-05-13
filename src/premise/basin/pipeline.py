from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .core import open_dataset, save_dataset
from .regrid import spatial_resample
from .spatial import clip_to_bbox, clip_with_vector
from .temporal import clip_time
from .variables import rename_variables


def _extract_summary(ds) -> dict[str, Any]:
    return {
        'dims': {k: int(v) for k, v in ds.sizes.items()},
        'coords': list(ds.coords),
        'data_vars': list(ds.data_vars),
    }


def process_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    bbox: dict[str, float] | None = None,
    vector_path: str | Path | None = None,
    region_field: str | None = None,
    region_values: str | list[str] | None = None,
    time_range: dict[str, str | None] | None = None,
    rename_map: dict[str, str] | None = None,
    target_resolution: float | None = None,
    target_grid_path: str | Path | None = None,
    resample_method: str = 'linear',
    lon_name: str = 'lon',
    lat_name: str = 'lat',
    time_name: str = 'time',
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    t0 = time.perf_counter()
    start_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    ds = open_dataset(input_path)
    before = _extract_summary(ds)

    if bbox is not None:
        ds = clip_to_bbox(ds, lon_name=lon_name, lat_name=lat_name, **bbox)

    if vector_path is not None:
        ds = clip_with_vector(
            ds,
            vector_path=vector_path,
            lon_name=lon_name,
            lat_name=lat_name,
            region_field=region_field,
            region_values=region_values,
            drop=True,
        )

    if time_range is not None:
        ds = clip_time(ds, start=time_range.get('start'), end=time_range.get('end'), time_name=time_name)

    if rename_map is not None:
        ds = rename_variables(ds, rename_map)

    if target_resolution is not None or target_grid_path is not None:
        ds = spatial_resample(
            ds,
            target_resolution=target_resolution,
            target_grid_path=target_grid_path,
            lon_name=lon_name,
            lat_name=lat_name,
            method=resample_method,
        )

    save_dataset(ds, output_path)
    after = _extract_summary(ds)
    elapsed = round(time.perf_counter() - t0, 4)

    report = {
        'input': str(input_path),
        'output': str(output_path),
        'start_time': start_dt,
        'elapsed_seconds': elapsed,
        'bbox': bbox,
        'vector_path': None if vector_path is None else str(vector_path),
        'region_field': region_field,
        'region_values': region_values,
        'time_range': time_range,
        'rename_map': rename_map,
        'target_resolution': target_resolution,
        'target_grid_path': None if target_grid_path is None else str(target_grid_path),
        'resample_method': resample_method,
        'before': before,
        'after': after,
    }

    if report_json is not None:
        rp = Path(report_json)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report
