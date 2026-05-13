from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str

try:
    import cdsapi
except ImportError:
    cdsapi = None


class ERA5LandHourlyDownloader(BaseDownloader):
    """
    ERA5-Land hourly downloader using CDS API.

    Current implementation targets:
    - dataset: reanalysis-era5-land
    - output: GRIB by default
    - minimal test case: single variable, single day, selected hours

    Requirements
    ------------
    1. cdsapi installed
    2. ~/.cdsapirc configured
    3. dataset Terms of Use accepted manually on CDS website
    """

    provider_name = "ERA5-Land"
    product_variant = "hourly"

    DATASET_NAME = "reanalysis-era5-land"

    def __init__(
        self,
        timeout: int = 300,
        max_retries: int = 2,
        sleep_seconds: float = 2.0,
        client: Optional["cdsapi.Client"] = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.client = client

    def download(self, request: DownloadRequest) -> list[Path]:
        if request.source_key.lower() != "era5land_hourly":
            raise ValueError(f"Unsupported source_key for ERA5LandHourlyDownloader: {request.source_key}")

        if not request.variables:
            raise ValueError("No variables were requested.")

        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")

        if request.start_date is None or request.end_date is None:
            raise ValueError("start_date and end_date must be provided.")

        if cdsapi is None and self.client is None:
            raise ImportError(
                "cdsapi is not installed. Please install it with: pip install 'cdsapi>=0.7.7'"
            )

        outdir = Path(request.target_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        request_id = make_request_id(request)

        years, months, days = self._expand_dates(request.start_date, request.end_date)

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
                "dataset_name": self.DATASET_NAME,
                "data_format": (request.format_preference or "grib").lower(),
            },
        )

        downloaded_files: list[Path] = []

        outname = self._build_output_filename(request)
        local_path = outdir / outname

        if local_path.exists() and local_path.stat().st_size > 0:
            downloaded_files.append(local_path)
            recorder.record_skipped_existing(
                url=f"cds://{self.DATASET_NAME}",
                local_path=str(local_path),
                size_bytes=local_path.stat().st_size,
            )
            recorder.finalize(outdir)
            return downloaded_files

        cds_request = {
            "variable": list(request.variables),
            "year": years,
            "month": months,
            "day": days,
            "time": self._build_default_times(),
            "data_format": (request.format_preference or "grib").lower(),
            "download_format": "unarchived",
        }

        # Optional area subset
        # CDS commonly expects [north, west, south, east]
        if request.bbox is not None:
            west, south, east, north = request.bbox
            cds_request["area"] = [north, west, south, east]

        client = self.client or cdsapi.Client(timeout=self.timeout, quiet=False, debug=False)

        success = False
        last_error = ""
        last_error_type = ""
        file_started_at = None
        file_finished_at = None
        file_elapsed = None

        for attempt in range(1, self.max_retries + 1):
            file_t0 = time.perf_counter()
            file_started_at = now_str()

            recorder.log_event(
                "INFO",
                f"dataset={self.DATASET_NAME} attempt={attempt} retrieval_started",
            )

            try:
                client.retrieve(self.DATASET_NAME, cds_request, str(local_path))

                file_elapsed = time.perf_counter() - file_t0
                file_finished_at = now_str()
                size_bytes = local_path.stat().st_size

                downloaded_files.append(local_path)
                recorder.record_success(
                    url=f"cds://{self.DATASET_NAME}",
                    local_path=str(local_path),
                    attempt=attempt,
                    download_started_at=file_started_at,
                    download_finished_at=file_finished_at,
                    elapsed_seconds=file_elapsed,
                    size_bytes=size_bytes,
                    http_status=None,
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
                    url=f"cds://{self.DATASET_NAME}",
                    local_path=str(local_path),
                    attempt=attempt,
                    download_started_at=file_started_at,
                    download_finished_at=file_finished_at,
                    elapsed_seconds=file_elapsed,
                    http_status=None,
                    error_type=last_error_type,
                    error_message=last_error,
                    file_exists_after_download=local_path.exists(),
                )

                if attempt < self.max_retries:
                    time.sleep(self.sleep_seconds)

        if not success:
            print(f"[FAILED] CDS request -> {last_error}")

        recorder.finalize(outdir)
        return downloaded_files

    @staticmethod
    def _expand_dates(start_date: str, end_date: str) -> tuple[list[str], list[str], list[str]]:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if end_dt < start_dt:
            raise ValueError("end_date must not be earlier than start_date")

        years = set()
        months = set()
        days = set()

        dt = start_dt
        while dt <= end_dt:
            years.add(f"{dt.year:04d}")
            months.add(f"{dt.month:02d}")
            days.add(f"{dt.day:02d}")
            dt += timedelta(days=1)

        return sorted(years), sorted(months), sorted(days)

    @staticmethod
    def _build_default_times() -> list[str]:
        # Full day hourly set
        return [f"{h:02d}:00" for h in range(24)]

    @staticmethod
    def _build_output_filename(request: DownloadRequest) -> str:
        start_part = request.start_date.replace("-", "") if request.start_date else "unknownstart"
        end_part = request.end_date.replace("-", "") if request.end_date else "unknownend"
        var_part = "-".join(request.variables)
        ext = "grib" if (request.format_preference or "grib").lower() == "grib" else "nc"
        return f"era5land_hourly_{var_part}_{start_part}_{end_part}.{ext}"