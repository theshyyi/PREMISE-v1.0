from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.catalog import DataSource
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str

try:
    import cdsapi
except ImportError:
    cdsapi = None


class CDSAPIDownloader(BaseDownloader):
    provider_name = "CDSAPI"

    def __init__(self, source_config: DataSource, timeout: int = 300, max_retries: int = 2,
                 sleep_seconds: float = 2.0, client: Optional["cdsapi.Client"] = None) -> None:
        self.source_config = source_config
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.client = client

    def download(self, request: DownloadRequest) -> list[Path]:
        src=self.source_config
        if request.target_dir is None: raise ValueError("target_dir must be provided.")
        if request.start_date is None or request.end_date is None: raise ValueError("start_date and end_date must be provided.")
        if cdsapi is None and self.client is None:
            raise ImportError("cdsapi is not installed. Please install it with: pip install 'cdsapi>=0.7.7'")
        outdir=Path(request.target_dir); outdir.mkdir(parents=True, exist_ok=True)
        request_id=make_request_id(request)
        params=src.params
        recorder = AcquisitionRunRecorder(request_id=request_id, source_key=request.source_key, provider_name=src.provider, product_variant=src.title, request_dict={"variables": list(request.variables), "start_date": request.start_date, "end_date": request.end_date, "bbox": request.bbox, "target_dir": request.target_dir, "notes": request.notes, "frequency": request.frequency, "format_preference": request.format_preference}, extra_metadata={"dataset_name": params['dataset_name'], "data_format": (request.format_preference or params.get('default_data_format', 'grib')).lower()})
        local_path=outdir / self._build_output_filename(request)
        downloaded=[]
        if local_path.exists() and local_path.stat().st_size>0:
            downloaded.append(local_path); recorder.record_skipped_existing(url=f"cds://{params['dataset_name']}", local_path=str(local_path), size_bytes=local_path.stat().st_size); recorder.finalize(outdir); return downloaded
        if params.get('time_mode') == 'hourly_all':
            years, months, days = self._expand_dates(request.start_date, request.end_date)
            cds_request = {"variable": list(request.variables), "year": years, "month": months, "day": days,
                           "time": [f"{h:02d}:00" for h in range(24)],
                           "data_format": (request.format_preference or params.get('default_data_format','grib')).lower(),
                           "download_format": params.get('default_download_format','unarchived')}
        else:
            raise ValueError(f"Unsupported CDS time_mode: {params.get('time_mode')}")
        if request.bbox is not None:
            west, south, east, north = request.bbox
            cds_request['area'] = [north, west, south, east]
        client = self.client or cdsapi.Client(timeout=self.timeout, quiet=False, debug=False)
        success=False; last_error=''; last_error_type=''
        for attempt in range(1, self.max_retries+1):
            tt=time.perf_counter(); started=now_str(); recorder.log_event('INFO', f"dataset={params['dataset_name']} attempt={attempt} retrieval_started")
            try:
                client.retrieve(params['dataset_name'], cds_request, str(local_path))
                elapsed=time.perf_counter()-tt; finished=now_str(); size_bytes=local_path.stat().st_size
                downloaded.append(local_path)
                recorder.record_success(url=f"cds://{params['dataset_name']}", local_path=str(local_path), attempt=attempt, download_started_at=started, download_finished_at=finished, elapsed_seconds=elapsed, size_bytes=size_bytes, http_status=None, file_exists_after_download=local_path.exists())
                success=True; break
            except Exception as e:
                elapsed=time.perf_counter()-tt; finished=now_str(); last_error=str(e); last_error_type=type(e).__name__
                if local_path.exists():
                    try: local_path.unlink()
                    except Exception: pass
                recorder.record_failure(url=f"cds://{params['dataset_name']}", local_path=str(local_path), attempt=attempt, download_started_at=started, download_finished_at=finished, elapsed_seconds=elapsed, http_status=None, error_type=last_error_type, error_message=last_error, file_exists_after_download=local_path.exists())
                if attempt < self.max_retries: time.sleep(self.sleep_seconds)
        if not success: print(f"[FAILED] CDS request -> {last_error}")
        recorder.finalize(outdir)
        return downloaded

    @staticmethod
    def _expand_dates(start_date: str, end_date: str) -> tuple[list[str], list[str], list[str]]:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if end_dt < start_dt: raise ValueError("end_date must not be earlier than start_date")
        years=set(); months=set(); days=set(); dt=start_dt
        while dt <= end_dt:
            years.add(f"{dt.year:04d}"); months.add(f"{dt.month:02d}"); days.add(f"{dt.day:02d}")
            dt += timedelta(days=1)
        return sorted(years), sorted(months), sorted(days)

    @staticmethod
    def _build_output_filename(request: DownloadRequest) -> str:
        start_part=request.start_date.replace('-', '') if request.start_date else 'unknownstart'
        end_part=request.end_date.replace('-', '') if request.end_date else 'unknownend'
        var_part='-'.join(request.variables)
        ext='grib' if (request.format_preference or 'grib').lower() == 'grib' else 'nc'
        return f"{request.source_key}_{var_part}_{start_part}_{end_part}.{ext}"
