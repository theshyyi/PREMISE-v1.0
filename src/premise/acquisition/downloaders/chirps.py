from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str


class CHIRPSDownloader(BaseDownloader):
    """
    CHIRPS downloader for PREMISE acquisition module.

    Current implementation targets:
    - CHIRPS v3
    - monthly global NetCDF by_year files
    """

    provider_name = "CHIRPS"
    product_variant = "monthly_global_netcdf_by_year"

    BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/netcdf/by_year/"
    FILENAME_PATTERN = "chirps-v3.0.{year}.monthly.nc"

    def __init__(
        self,
        timeout: int = 120,
        chunk_size: int = 1024 * 1024,
        max_retries: int = 3,
        sleep_seconds: float = 1.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()

    def download(self, request: DownloadRequest) -> list[Path]:
        if request.source_key.lower() != "chirps_monthly_nc":
            raise ValueError(f"Unsupported source_key for CHIRPSDownloader: {request.source_key}")

        if not request.variables:
            raise ValueError("No variables were requested.")

        vars_lower = {v.lower() for v in request.variables}
        if "precipitation" not in vars_lower:
            raise ValueError("CHIRPSDownloader currently supports only precipitation.")

        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")

        if request.start_date is None or request.end_date is None:
            raise ValueError("start_date and end_date must be provided.")

        outdir = Path(request.target_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        request_id = make_request_id(request)
        years = self._expand_years(request.start_date, request.end_date)

        recorder = AcquisitionRunRecorder(
            request_id=request_id,
            source_key=request.source_key,
            provider_name=self.provider_name,
            product_variant=self.product_variant,
            request_dict={
                "variables": list(request.variables),
                "start_date": request.start_date,
                "end_date": request.end_date,
                "bbox": request.bbox,
                "target_dir": request.target_dir,
                "notes": request.notes,
                "frequency": request.frequency,
                "format_preference": request.format_preference,
            },
            extra_metadata={
                "base_url": self.BASE_URL,
                "filename_pattern": self.FILENAME_PATTERN,
            },
        )

        downloaded_files: list[Path] = []

        for year in years:
            filename = self.FILENAME_PATTERN.format(year=year)
            url = urljoin(self.BASE_URL, filename)
            local_path = outdir / filename

            if local_path.exists() and local_path.stat().st_size > 0:
                downloaded_files.append(local_path)
                recorder.record_skipped_existing(
                    url=url,
                    local_path=str(local_path),
                    size_bytes=local_path.stat().st_size,
                )
                continue

            success = False
            last_error = ""
            last_error_type = ""
            last_http_status = None

            for attempt in range(1, self.max_retries + 1):
                file_t0 = time.perf_counter()
                file_started_at = now_str()

                recorder.log_event(
                    "INFO",
                    f"url={url} attempt={attempt} download_started",
                )

                try:
                    with self.session.get(url, stream=True, timeout=self.timeout) as r:
                        last_http_status = r.status_code
                        r.raise_for_status()

                        with open(local_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=self.chunk_size):
                                if chunk:
                                    f.write(chunk)

                    file_elapsed = time.perf_counter() - file_t0
                    file_finished_at = now_str()
                    size_bytes = local_path.stat().st_size

                    downloaded_files.append(local_path)
                    recorder.record_success(
                        url=url,
                        local_path=str(local_path),
                        attempt=attempt,
                        download_started_at=file_started_at,
                        download_finished_at=file_finished_at,
                        elapsed_seconds=file_elapsed,
                        size_bytes=size_bytes,
                        http_status=last_http_status,
                        file_exists_after_download=local_path.exists(),
                    )
                    success = True
                    break

                except Exception as e:
                    file_elapsed = time.perf_counter() - file_t0
                    file_finished_at = now_str()
                    last_error = str(e)
                    last_error_type = type(e).__name__

                    if local_path.exists():
                        try:
                            local_path.unlink()
                        except Exception:
                            pass

                    recorder.record_failure(
                        url=url,
                        local_path=str(local_path),
                        attempt=attempt,
                        download_started_at=file_started_at,
                        download_finished_at=file_finished_at,
                        elapsed_seconds=file_elapsed,
                        http_status=last_http_status,
                        error_type=last_error_type,
                        error_message=last_error,
                        file_exists_after_download=local_path.exists(),
                    )

                    if attempt < self.max_retries:
                        time.sleep(self.sleep_seconds)

            if not success:
                print(f"[FAILED] {url} -> {last_error}")

        recorder.finalize(outdir)
        return downloaded_files

    @staticmethod
    def _expand_years(start_date: str, end_date: str) -> list[int]:
        ys = int(start_date.split("-")[0])
        ye = int(end_date.split("-")[0])
        return list(range(ys, ye + 1))