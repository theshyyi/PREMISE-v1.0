from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from premise.acquisition.base import DownloadRequest
from premise.acquisition.router import download_dataset
from premise.basin.pipeline import process_dataset
from premise.conversion.api import detect_format, convert_to_netcdf

from .common import ensure_dir, normalize_bbox, save_json


NETCDF_SUFFIXES = {'.nc', '.nc4'}


def _merge_netcdf_files(nc_files: list[Path], out_nc: Path) -> Path:
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    if len(nc_files) == 1:
        return nc_files[0]
    ds = xr.open_mfdataset([str(p) for p in nc_files], combine='by_coords')
    try:
        ds.to_netcdf(out_nc)
    finally:
        ds.close()
    return out_nc


def _convert_downloads_to_nc(
    files: list[Path],
    *,
    out_dir: Path,
    source_key: str,
    conversion_hints: dict[str, Any] | None = None,
) -> Path:
    if not files:
        raise FileNotFoundError(f'No downloaded files found for source: {source_key}')

    files = [Path(f) for f in files]
    nc_files = [p for p in files if p.suffix.lower() in NETCDF_SUFFIXES]
    if len(nc_files) == len(files):
        return _merge_netcdf_files(nc_files, out_dir / f'{source_key}_merged.nc')

    fmt = detect_format(files[0])
    conversion_hints = conversion_hints or {}

    if fmt == 'geotiff':
        # stack the folder or use the parent folder for time-aware tif collections
        geotiff_root = files[0].parent if len(files) > 1 else files[0]
        out_nc = out_dir / f'{source_key}_converted.nc'
        convert_to_netcdf(geotiff_root, out_nc, hints=conversion_hints)
        return out_nc

    piece_dir = ensure_dir(out_dir / 'pieces')
    converted: list[Path] = []
    for idx, src in enumerate(files, start=1):
        out_nc = piece_dir / f'{source_key}_{idx:04d}.nc'
        convert_to_netcdf(src, out_nc, hints=conversion_hints)
        converted.append(out_nc)

    return _merge_netcdf_files(converted, out_dir / f'{source_key}_converted_merged.nc')


def _write_builder_markdown(report: dict[str, Any], out_path: Path) -> None:
    lines = ['# Hydro data builder report', '']
    lines.append(f"- task_name: {report.get('task_name', '')}")
    lines.append(f"- status: {report.get('status', '')}")
    lines.append(f"- output_dir: {report.get('output_dir', '')}")
    if report.get('merged_output'):
        lines.append(f"- merged_output: {report['merged_output']}")
    lines.append('')
    lines.append('## Requests')
    for item in report.get('requests', []):
        lines.append(f"### {item.get('source_key', '')}")
        lines.append(f"- status: {item.get('status', '')}")
        lines.append(f"- downloaded_count: {item.get('downloaded_count', 0)}")
        lines.append(f"- raw_dir: {item.get('raw_dir', '')}")
        lines.append(f"- converted_nc: {item.get('converted_nc', '')}")
        lines.append(f"- prepared_nc: {item.get('prepared_nc', '')}")
        if item.get('error_message'):
            lines.append(f"- error_message: {item['error_message']}")
        lines.append('')
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def run_hydro_data_builder_task(task: dict[str, Any]) -> dict[str, Any]:
    task_name = str(task.get('name', 'hydro_data_builder'))
    output_dir = ensure_dir(task['output_dir'])
    raw_root = ensure_dir(output_dir / 'raw_download')
    converted_root = ensure_dir(output_dir / 'converted_nc')
    prepared_root = ensure_dir(output_dir / 'final_ready')
    report_root = ensure_dir(output_dir / 'reports')

    time_range = task.get('time_range', {}) or {}
    start_date = time_range.get('start')
    end_date = time_range.get('end')
    bbox = normalize_bbox(task.get('bbox'))
    region = task.get('region', {}) or {}

    requests = task.get('requests')
    if not requests:
        requests = [{
            'source_key': task['source_key'],
            'variables': task['variables'],
            'rename_map': task.get('rename_map'),
            'conversion_hints': task.get('conversion_hints'),
        }]

    records: list[dict[str, Any]] = []
    prepared_products: list[Path] = []

    for req_cfg in requests:
        source_key = req_cfg['source_key']
        variables = tuple(req_cfg.get('variables', []))
        raw_dir = ensure_dir(raw_root / source_key)
        conv_dir = ensure_dir(converted_root / source_key)
        req_record = {
            'source_key': source_key,
            'variables': list(variables),
            'status': 'FAILED',
            'raw_dir': str(raw_dir),
            'downloaded_count': 0,
            'downloaded_files': [],
            'converted_nc': '',
            'prepared_nc': '',
            'error_message': '',
        }
        try:
            request_bbox = bbox if bool(req_cfg.get('pass_bbox_to_downloader', task.get('pass_bbox_to_downloader', False))) else None
            request = DownloadRequest(
                source_key=source_key,
                variables=variables,
                start_date=start_date,
                end_date=end_date,
                bbox=None if request_bbox is None else (
                    request_bbox['min_lon'], request_bbox['min_lat'], request_bbox['max_lon'], request_bbox['max_lat']
                ),
                target_dir=str(raw_dir),
                notes=req_cfg.get('notes', ''),
                frequency=req_cfg.get('frequency'),
                format_preference=req_cfg.get('format_preference'),
            )
            downloaded = [Path(p) for p in download_dataset(request, **(req_cfg.get('downloader_kwargs', {}) or {}))]
            req_record['downloaded_count'] = len(downloaded)
            req_record['downloaded_files'] = [str(p) for p in downloaded]
            if not downloaded:
                req_record['status'] = 'NO_FILES'
                records.append(req_record)
                continue

            converted_nc = _convert_downloads_to_nc(
                downloaded,
                out_dir=conv_dir,
                source_key=source_key,
                conversion_hints=req_cfg.get('conversion_hints'),
            )
            req_record['converted_nc'] = str(converted_nc)

            prepared_nc = prepared_root / f'{source_key}_prepared.nc'
            process_dataset(
                converted_nc,
                prepared_nc,
                bbox=bbox,
                vector_path=region.get('path'),
                region_field=region.get('field'),
                region_values=region.get('values'),
                time_range={'start': start_date, 'end': end_date},
                rename_map=req_cfg.get('rename_map'),
                target_resolution=task.get('target_resolution'),
                target_grid_path=task.get('target_grid_path'),
                resample_method=task.get('resample_method', 'linear'),
                report_json=report_root / f'{source_key}_process_report.json',
            )
            req_record['prepared_nc'] = str(prepared_nc)
            req_record['status'] = 'SUCCESS'
            prepared_products.append(prepared_nc)
        except Exception as e:
            req_record['error_message'] = f'{type(e).__name__}: {e}'
        records.append(req_record)

    merged_output = ''
    merged_status = 'PARTIAL' if any(r['status'] == 'SUCCESS' for r in records) else 'FAILED'
    if prepared_products and task.get('merge_prepared', True):
        merge_name = task.get('merged_filename', 'hydro_model_input.nc')
        merged_path = prepared_root / merge_name
        datasets = [xr.open_dataset(p) for p in prepared_products]
        try:
            merged = xr.merge(datasets, compat='override', join='outer')
            try:
                merged.to_netcdf(merged_path)
            finally:
                merged.close()
        finally:
            for ds in datasets:
                ds.close()
        merged_output = str(merged_path)
        merged_status = 'SUCCESS'

    report = {
        'task_name': task_name,
        'status': merged_status,
        'output_dir': str(output_dir),
        'merged_output': merged_output,
        'requests': records,
    }
    save_json(report, report_root / f'{task_name}_summary.json')
    _write_builder_markdown(report, report_root / f'{task_name}_report.md')
    return report


def run_hydro_data_builder_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_hydro_data_builder_task(task) for task in tasks if task.get('enabled', True)]
