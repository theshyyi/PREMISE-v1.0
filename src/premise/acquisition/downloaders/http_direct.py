from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.catalog import DataSource
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str


class HTTPDirectDownloader(BaseDownloader):
    """
    Generic downloader for datasets with deterministic direct-file URLs.

    Supported time_expansion values
    -------------------------------
    year
        One file per calendar year.
    date
        One file per calendar date.
    none
        A single static file regardless of the requested period.
    decade
        One file per multi-year block. The block length and anchor year can be
        controlled with ``decade_length`` and ``decade_anchor_year`` in the
        source configuration params.
    """

    provider_name = "HTTPDirect"

    def __init__(
        self,
        source_config: DataSource,
        timeout: int = 120,
        chunk_size: int = 1024 * 1024,
        max_retries: int = 3,
        sleep_seconds: float = 1.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.source_config = source_config
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()

    def download(self, request: DownloadRequest) -> list[Path]:
        if request.source_key.lower() != self.source_config.key.lower():
            raise ValueError(
                f"HTTPDirectDownloader received mismatched source_key: {request.source_key} "
                f"!= {self.source_config.key}"
            )

        if not request.variables:
            raise ValueError("No variables were requested.")

        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")

        if request.start_date is None or request.end_date is None:
            raise ValueError("start_date and end_date must be provided.")

        outdir = Path(request.target_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        params = self.source_config.params
        base_url = params["base_url"]
        filename_pattern = params["filename_pattern"]
        directory_pattern = params.get("directory_pattern", "")
        time_expansion = params.get("time_expansion", "year")

        request_id = make_request_id(request)

        recorder = AcquisitionRunRecorder(
            request_id=request_id,
            source_key=request.source_key,
            provider_name=self.source_config.provider,
            product_variant=self.source_config.title,
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
                "base_url": base_url,
                "filename_pattern": filename_pattern,
                "directory_pattern": directory_pattern,
                "time_expansion": time_expansion,
            },
        )

        downloaded_files: list[Path] = []

        for ctx in self._expand_contexts(
            start_date=request.start_date,
            end_date=request.end_date,
            time_expansion=time_expansion,
            params=params,
        ):
            filename = filename_pattern.format(**ctx)
            subdir = directory_pattern.format(**ctx) if directory_pattern else ""
            url = urljoin(base_url, f"{subdir}{filename}")
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

    @classmethod
    def _expand_contexts(
        cls,
        *,
        start_date: str,
        end_date: str,
        time_expansion: str,
        params: dict,
    ) -> list[dict]:
        time_expansion = time_expansion.lower().strip()
        if time_expansion == "year":
            return [{"year": year} for year in cls._expand_years(start_date, end_date)]
        if time_expansion == "date":
            contexts = []
            for dt in cls._expand_dates(start_date, end_date):
                contexts.append(
                    {
                        "year": f"{dt.year:04d}",
                        "month": f"{dt.month:02d}",
                        "day": f"{dt.day:02d}",
                        "date": f"{dt.year:04d}{dt.month:02d}{dt.day:02d}",
                    }
                )
            return contexts
        if time_expansion in {"none", "static", "single"}:
            return [{}]
        if time_expansion == "decade":
            anchor = int(params.get("decade_anchor_year", 1901))
            length = int(params.get("decade_length", 10))
            return [
                {
                    "start_year": f"{start_year:04d}",
                    "end_year": f"{end_year:04d}",
                    "period_start": f"{start_year:04d}",
                    "period_end": f"{end_year:04d}",
                }
                for start_year, end_year in cls._expand_year_blocks(start_date, end_date, anchor=anchor, length=length)
            ]
        raise ValueError(f"Unsupported time_expansion: {time_expansion}")

    @staticmethod
    def _expand_years(start_date: str, end_date: str) -> list[int]:
        ys = int(start_date.split("-")[0])
        ye = int(end_date.split("-")[0])
        if ye < ys:
            raise ValueError("end_date must not be earlier than start_date")
        return list(range(ys, ye + 1))

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

    @staticmethod
    def _expand_year_blocks(start_date: str, end_date: str, *, anchor: int, length: int) -> list[tuple[int, int]]:
        if length <= 0:
            raise ValueError("length must be positive")
        ys = int(start_date.split("-")[0])
        ye = int(end_date.split("-")[0])
        if ye < ys:
            raise ValueError("end_date must not be earlier than start_date")

        blocks: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for year in range(ys, ye + 1):
            block_start = anchor + ((year - anchor) // length) * length
            block_end = block_start + length - 1
            block = (block_start, block_end)
            if block not in seen:
                blocks.append(block)
                seen.add(block)
        return blocks
