from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str


class IMERGFinalHDF5Downloader(BaseDownloader):
    """
    IMERG Final Run HDF5 downloader.

    Current implementation
    ----------------------
    - Targets PPS archive-style daily directory listings
    - Looks for HDF5 files under:
        https://arthurhouhttps.pps.eosdis.nasa.gov/gpmdata/YYYY/MM/DD/imerg/
    - Uses HTML directory parsing to identify HDF5 files
    - Can limit file count for smoke testing

    Notes
    -----
    1. IMERG file naming/version may change over time (e.g., V07B).
    2. This downloader intentionally parses available files from the directory
       rather than hard-coding the exact version suffix.
    3. Depending on the current server-side access configuration, authenticated
       HTTPS/session handling may be required.
    """

    provider_name = "IMERG"
    product_variant = "final_hdf5"

    BASE_URL = "https://arthurhouhttps.pps.eosdis.nasa.gov/gpmdata/"

    # broad pattern to capture IMERG HDF5 files from listing
    HDF5_REGEX = re.compile(r'href="([^"]+\.HDF5)"', re.IGNORECASE)

    def __init__(
        self,
        timeout: int = 120,
        chunk_size: int = 1024 * 1024,
        max_retries: int = 3,
        sleep_seconds: float = 1.0,
        session: Optional[requests.Session] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        file_limit: Optional[int] = 1,
    ) -> None:
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()
        self.file_limit = file_limit

        # Optional auth; if your access route requires a different auth mode,
        # adapt the session externally and pass it in.
        if username is not None and password is not None:
            self.session.auth = (username, password)

    def download(self, request: DownloadRequest) -> list[Path]:
        if request.source_key.lower() != "imerg_final_hdf5":
            raise ValueError(f"Unsupported source_key for IMERGFinalHDF5Downloader: {request.source_key}")

        if not request.variables:
            raise ValueError("No variables were requested.")

        vars_lower = {v.lower() for v in request.variables}
        if "precipitation" not in vars_lower:
            raise ValueError("IMERGFinalHDF5Downloader currently supports only precipitation.")

        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")

        if request.start_date is None or request.end_date is None:
            raise ValueError("start_date and end_date must be provided.")

        outdir = Path(request.target_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        request_id = make_request_id(request)
        dates = self._expand_dates(request.start_date, request.end_date)

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
                "file_limit": self.file_limit,
            },
        )

        downloaded_files: list[Path] = []

        for dt in dates:
            year = f"{dt.year:04d}"
            month = f"{dt.month:02d}"
            day = f"{dt.day:02d}"

            directory_url = urljoin(self.BASE_URL, f"{year}/{month}/{day}/imerg/")

            recorder.log_event("INFO", f"directory_listing_started url={directory_url}")

            try:
                listing_files = self._list_hdf5_files(directory_url)
            except Exception as e:
                recorder.record_failure(
                    url=directory_url,
                    local_path="",
                    attempt=1,
                    download_started_at=now_str(),
                    download_finished_at=now_str(),
                    elapsed_seconds=0.0,
                    http_status=None,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    file_exists_after_download=False,
                )
                print(f"[FAILED] listing {directory_url} -> {e}")
                continue

            if self.file_limit is not None:
                listing_files = listing_files[: self.file_limit]

            for fname in listing_files:
                file_url = urljoin(directory_url, fname)
                local_path = outdir / fname

                if local_path.exists() and local_path.stat().st_size > 0:
                    downloaded_files.append(local_path)
                    recorder.record_skipped_existing(
                        url=file_url,
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
                        f"url={file_url} attempt={attempt} download_started",
                    )

                    try:
                        with self.session.get(file_url, stream=True, timeout=self.timeout) as r:
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
                            url=file_url,
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
                            url=file_url,
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
                    print(f"[FAILED] {file_url} -> {last_error}")

        recorder.finalize(outdir)
        return downloaded_files

    def _list_hdf5_files(self, directory_url: str) -> list[str]:
        r = self.session.get(directory_url, timeout=self.timeout)
        r.raise_for_status()
        matches = self.HDF5_REGEX.findall(r.text)

        # keep only IMERG-like HDF5 products
        files = []
        for m in matches:
            name = m.split("/")[-1]
            if "3IMERG" in name and name.upper().endswith(".HDF5"):
                files.append(name)

        # remove duplicates while preserving order
        seen = set()
        out = []
        for f in files:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    @staticmethod
    def _expand_dates(start_date: str, end_date: str) -> list[datetime]:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if end_dt < start_dt:
            raise ValueError("end_date must not be earlier than start_date")

        out = []
        dt = start_dt
        while dt <= end_dt:
            out.append(dt)
            dt += timedelta(days=1)
        return out