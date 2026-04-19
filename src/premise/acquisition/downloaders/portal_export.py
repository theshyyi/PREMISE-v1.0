from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import requests

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.catalog import DataSource
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str


class PortalExportDownloader(BaseDownloader):
    provider_name = "PortalExport"

    FAMILY_PRODUCT_MAP = {
        "persiann_portal": "PERSIANN",
        "persiann_ccs_portal": "PERSIANN-CCS",
        "persiann_cdr_portal": "PERSIANN-CDR",
        "persiann_ccs_cdr_v2_portal": "PERSIANN-CCS-CDR V2.0",
        "pdir_now_portal": "PDIR-Now",
        "persiann_v3_portal": "PERSIANN V3",
        "persiann_cdr_v3_portal": "PERSIANN-CDR V3",
        "cmfd": "CMFD",
    }

    def __init__(self, source_config: DataSource, export_url: Optional[str] = None, timeout: int = 120,
                 chunk_size: int = 1024 * 1024, max_retries: int = 3, sleep_seconds: float = 1.0,
                 session: Optional[requests.Session] = None) -> None:
        self.source_config = source_config
        self.export_url = export_url
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()

    def download(self, request: DownloadRequest) -> list[Path]:
        src=self.source_config
        if request.target_dir is None: raise ValueError("target_dir must be provided.")
        if not self.export_url: raise ValueError("PortalExportDownloader requires export_url.")
        outdir=Path(request.target_dir); outdir.mkdir(parents=True, exist_ok=True)
        request_id=make_request_id(request)
        family_product = self.FAMILY_PRODUCT_MAP.get(src.key, src.title)
        local_path = outdir / self._guess_output_filename(request)
        recorder=AcquisitionRunRecorder(request_id=request_id, source_key=request.source_key, provider_name=src.provider, product_variant=src.title, request_dict={"variables": list(request.variables), "start_date": request.start_date, "end_date": request.end_date, "bbox": request.bbox, "target_dir": request.target_dir, "notes": request.notes, "frequency": request.frequency, "format_preference": request.format_preference}, extra_metadata={"family_product": family_product, "export_url": self.export_url})
        files=[]
        if local_path.exists() and local_path.stat().st_size>0:
            files.append(local_path); recorder.record_skipped_existing(url=self.export_url, local_path=str(local_path), size_bytes=local_path.stat().st_size); recorder.finalize(outdir); return files
        success=False; last_error=''; last_error_type=''; last_http_status=None
        for attempt in range(1, self.max_retries+1):
            tt=time.perf_counter(); started=now_str(); recorder.log_event('INFO', f'family_product={family_product} url={self.export_url} attempt={attempt} download_started')
            try:
                with self.session.get(self.export_url, stream=True, timeout=self.timeout) as r:
                    last_http_status=r.status_code; r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=self.chunk_size):
                            if chunk: f.write(chunk)
                elapsed=time.perf_counter()-tt; finished=now_str(); size_bytes=local_path.stat().st_size
                files.append(local_path)
                recorder.record_success(url=self.export_url, local_path=str(local_path), attempt=attempt, download_started_at=started, download_finished_at=finished, elapsed_seconds=elapsed, size_bytes=size_bytes, http_status=last_http_status, file_exists_after_download=local_path.exists())
                success=True; break
            except Exception as e:
                elapsed=time.perf_counter()-tt; finished=now_str(); last_error=str(e); last_error_type=type(e).__name__
                if local_path.exists():
                    try: local_path.unlink()
                    except Exception: pass
                recorder.record_failure(url=self.export_url, local_path=str(local_path), attempt=attempt, download_started_at=started, download_finished_at=finished, elapsed_seconds=elapsed, http_status=last_http_status, error_type=last_error_type, error_message=last_error, file_exists_after_download=local_path.exists())
                if attempt < self.max_retries: time.sleep(self.sleep_seconds)
        if not success: print(f"[FAILED] {self.export_url} -> {last_error}")
        recorder.finalize(outdir)
        return files

    @staticmethod
    def _guess_output_filename(request: DownloadRequest) -> str:
        fmt = (request.format_preference or 'nc').lower()
        if 'netcdf' in fmt or fmt == 'nc': ext='nc'
        elif 'tif' in fmt or 'geotiff' in fmt: ext='tif'
        elif 'arc' in fmt or 'grid' in fmt: ext='asc'
        else: ext='dat'
        source_part=request.source_key.lower(); start_part=request.start_date.replace('-', '') if request.start_date else 'unknownstart'; end_part=request.end_date.replace('-', '') if request.end_date else 'unknownend'; var_part='-'.join(request.variables) if request.variables else 'precipitation'
        return f"{source_part}_{var_part}_{start_part}_{end_part}.{ext}"
