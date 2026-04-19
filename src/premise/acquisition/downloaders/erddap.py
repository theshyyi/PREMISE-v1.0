from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.catalog import DataSource
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str


class ERDDAPDownloader(BaseDownloader):
    provider_name = "ERDDAP"

    def __init__(self, source_config: DataSource, timeout: int = 120, chunk_size: int = 1024 * 1024,
                 max_retries: int = 3, sleep_seconds: float = 1.0, session: Optional[requests.Session] = None) -> None:
        self.source_config = source_config
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()

    def download(self, request: DownloadRequest) -> list[Path]:
        src = self.source_config
        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")
        if request.start_date is None or request.end_date is None:
            raise ValueError("start_date and end_date must be provided.")
        outdir = Path(request.target_dir); outdir.mkdir(parents=True, exist_ok=True)
        request_id = make_request_id(request)
        params = src.params
        if request.bbox is None:
            west, south, east, north = params.get('default_bbox', (0.125, -59.875, 359.875, 59.875))
        else:
            west, south, east, north = request.bbox
        var = params.get('query_variable', request.variables[0])
        t0 = f"{request.start_date}T00:00:00Z"; t1 = f"{request.end_date}T00:00:00Z"
        query = f"{var}[({t0}):{params.get('time_step',1)}:({t1})][({south}):{params.get('lat_step',1)}:({north})][({west}):{params.get('lon_step',1)}:({east})]"
        url = f"{params['base_url']}?{quote(query, safe='[]():,')}"
        local_path = outdir / f"{request.source_key}_{request.start_date.replace('-', '')}_{request.end_date.replace('-', '')}.nc"
        recorder = AcquisitionRunRecorder(request_id=request_id, source_key=request.source_key, provider_name=src.provider, product_variant=src.title, request_dict={"variables": list(request.variables), "start_date": request.start_date, "end_date": request.end_date, "bbox": request.bbox, "target_dir": request.target_dir, "notes": request.notes, "frequency": request.frequency, "format_preference": request.format_preference}, extra_metadata={"request_url": url})
        downloaded=[]
        if local_path.exists() and local_path.stat().st_size > 0:
            downloaded.append(local_path); recorder.record_skipped_existing(url=url, local_path=str(local_path), size_bytes=local_path.stat().st_size); recorder.finalize(outdir); return downloaded
        success=False; last_error=''; last_error_type=''; last_http_status=None
        for attempt in range(1, self.max_retries+1):
            tt=time.perf_counter(); started=now_str(); recorder.log_event('INFO', f'url={url} attempt={attempt} download_started')
            try:
                with self.session.get(url, stream=True, timeout=self.timeout) as r:
                    last_http_status=r.status_code; r.raise_for_status()
                    with open(local_path,'wb') as f:
                        for chunk in r.iter_content(chunk_size=self.chunk_size):
                            if chunk: f.write(chunk)
                elapsed=time.perf_counter()-tt; finished=now_str(); size_bytes=local_path.stat().st_size
                downloaded.append(local_path)
                recorder.record_success(url=url, local_path=str(local_path), attempt=attempt, download_started_at=started, download_finished_at=finished, elapsed_seconds=elapsed, size_bytes=size_bytes, http_status=last_http_status, file_exists_after_download=local_path.exists())
                success=True; break
            except Exception as e:
                elapsed=time.perf_counter()-tt; finished=now_str(); last_error=str(e); last_error_type=type(e).__name__
                if local_path.exists():
                    try: local_path.unlink()
                    except Exception: pass
                recorder.record_failure(url=url, local_path=str(local_path), attempt=attempt, download_started_at=started, download_finished_at=finished, elapsed_seconds=elapsed, http_status=last_http_status, error_type=last_error_type, error_message=last_error, file_exists_after_download=local_path.exists())
                if attempt < self.max_retries: time.sleep(self.sleep_seconds)
        if not success: print(f"[FAILED] {url} -> {last_error}")
        recorder.finalize(outdir)
        return downloaded
