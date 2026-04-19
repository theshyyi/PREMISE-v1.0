from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.catalog import DataSource
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str


class HTTPListingDownloader(BaseDownloader):
    provider_name = "HTTPListing"

    def __init__(
        self,
        source_config: DataSource,
        timeout: int = 120,
        chunk_size: int = 1024 * 1024,
        max_retries: int = 3,
        sleep_seconds: float = 1.0,
        session: Optional[requests.Session] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        file_limit: Optional[int] = None,
    ) -> None:
        self.source_config = source_config
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()
        if username is not None and password is not None:
            self.session.auth = (username, password)
        self.file_limit = file_limit

    def download(self, request: DownloadRequest) -> list[Path]:
        src = self.source_config
        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")
        if request.start_date is None or request.end_date is None:
            raise ValueError("start_date and end_date must be provided.")

        outdir = Path(request.target_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        request_id = make_request_id(request)
        params = src.params

        regex = re.compile(params.get("file_regex", r'href="([^"]+)"'), re.IGNORECASE)
        file_limit = self.file_limit if self.file_limit is not None else params.get("file_limit_default")
        time_expansion = params.get("time_expansion", "date")

        recorder = AcquisitionRunRecorder(
            request_id=request_id,
            source_key=request.source_key,
            provider_name=src.provider,
            product_variant=src.title,
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
                "base_url": params["base_url"],
                "directory_pattern": params.get("directory_pattern", ""),
                "file_regex": params.get("file_regex"),
                "time_expansion": time_expansion,
            },
        )

        discovered: list[tuple[str, str]] = []
        seen_urls: set[str] = set()

        for ctx in self._expand_contexts(request.start_date, request.end_date, time_expansion=time_expansion):
            directory_url = urljoin(
                params["base_url"],
                (params.get("directory_pattern", "") or "").format(**ctx),
            )
            recorder.log_event("INFO", f"directory_listing_started url={directory_url}")

            try:
                with self.session.get(directory_url, timeout=self.timeout) as r:
                    r.raise_for_status()
                    matches = regex.findall(r.text)
            except Exception as e:
                recorder.record_failure(
                    url=directory_url,
                    local_path="",
                    attempt=1,
                    download_started_at=now_str(),
                    download_finished_at=now_str(),
                    elapsed_seconds=0.0,
                    http_status=getattr(getattr(e, "response", None), "status_code", None),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    file_exists_after_download=False,
                )
                print(f"[FAILED] listing {directory_url} -> {e}")
                continue

            candidates: list[str] = []
            for match in matches:
                if isinstance(match, tuple):
                    raw_name = next((item for item in match if item), "")
                else:
                    raw_name = match
                name = raw_name.split("/")[-1].strip()
                if not name or name in {"../", "./"}:
                    continue
                candidates.append(name)

            filtered = self._filter_filenames(
                candidates,
                request=request,
                params=params,
                ctx=ctx,
            )

            for fname in filtered:
                file_url = urljoin(directory_url if directory_url.endswith("/") else directory_url + "/", fname)
                if file_url in seen_urls:
                    continue
                seen_urls.add(file_url)
                discovered.append((file_url, fname))

        if file_limit is not None:
            discovered = discovered[:file_limit]

        downloaded: list[Path] = []
        for file_url, fname in discovered:
            local_path = outdir / Path(fname).name
            if local_path.exists() and local_path.stat().st_size > 0:
                downloaded.append(local_path)
                recorder.record_skipped_existing(url=file_url, local_path=str(local_path), size_bytes=local_path.stat().st_size)
                continue

            success = False
            last_error = ""
            last_error_type = ""
            last_http_status = None
            for attempt in range(1, self.max_retries + 1):
                t0 = time.perf_counter()
                started = now_str()
                recorder.log_event("INFO", f"url={file_url} attempt={attempt} download_started")
                try:
                    with self.session.get(file_url, stream=True, timeout=self.timeout) as r:
                        last_http_status = r.status_code
                        r.raise_for_status()
                        with open(local_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=self.chunk_size):
                                if chunk:
                                    f.write(chunk)
                    elapsed = time.perf_counter() - t0
                    finished = now_str()
                    size_bytes = local_path.stat().st_size
                    downloaded.append(local_path)
                    recorder.record_success(
                        url=file_url,
                        local_path=str(local_path),
                        attempt=attempt,
                        download_started_at=started,
                        download_finished_at=finished,
                        elapsed_seconds=elapsed,
                        size_bytes=size_bytes,
                        http_status=last_http_status,
                        file_exists_after_download=local_path.exists(),
                    )
                    success = True
                    break
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    finished = now_str()
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
                        download_started_at=started,
                        download_finished_at=finished,
                        elapsed_seconds=elapsed,
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
        return downloaded

    def _filter_filenames(self, filenames: list[str], *, request: DownloadRequest, params: dict, ctx: dict) -> list[str]:
        contains_cfg = params.get("contains", [])
        contains = [c.lower() for c in contains_cfg]

        contains_template_cfg = params.get("contains_template", [])
        if isinstance(contains_template_cfg, str):
            contains_template_cfg = [contains_template_cfg]
        formatted_contains = [
            token.format(**ctx).lower()
            for token in contains_template_cfg
        ]

        out: list[str] = []
        seen: set[str] = set()
        for name in filenames:
            lower_name = name.lower()
            if contains and not all(token in lower_name for token in contains):
                continue
            if formatted_contains and not all(token in lower_name for token in formatted_contains):
                continue
            if not self._filename_in_requested_window(name=name, start_date=request.start_date, end_date=request.end_date, params=params):
                continue
            if name not in seen:
                out.append(name)
                seen.add(name)
        return out

    @classmethod
    def _filename_in_requested_window(cls, *, name: str, start_date: str, end_date: str, params: dict) -> bool:
        date_regex = params.get("filename_date_regex")
        if not date_regex:
            return True
        match = re.search(date_regex, name)
        if not match:
            return False

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        file_dt = cls._coerce_match_to_datetime(match)
        return start_dt <= file_dt <= end_dt

    @staticmethod
    def _coerce_match_to_datetime(match: re.Match) -> datetime:
        gd = match.groupdict()
        if "date" in gd and gd["date"]:
            token = gd["date"]
            if len(token) == 6:
                return datetime.strptime(token, "%Y%m")
            if len(token) == 8:
                return datetime.strptime(token, "%Y%m%d")
            if len(token) == 10:
                return datetime.strptime(token, "%Y%m%d%H")
            if len(token) == 12:
                return datetime.strptime(token, "%Y%m%d%H%M")
        if "yearmonth" in gd and gd["yearmonth"]:
            return datetime.strptime(gd["yearmonth"], "%Y%m")

        year = int(gd.get("year") or 1900)
        month = int(gd.get("month") or 1)
        day = int(gd.get("day") or 1)
        hour = int(gd.get("hour") or 0)
        minute = int(gd.get("minute") or 0)
        return datetime(year, month, day, hour, minute)

    @classmethod
    def _expand_contexts(cls, start_date: str, end_date: str, *, time_expansion: str) -> list[dict]:
        time_expansion = time_expansion.lower().strip()
        if time_expansion == "date":
            return [
                {
                    "year": f"{dt.year:04d}",
                    "month": f"{dt.month:02d}",
                    "day": f"{dt.day:02d}",
                    "date": f"{dt.year:04d}{dt.month:02d}{dt.day:02d}",
                }
                for dt in cls._expand_dates(start_date, end_date)
            ]
        if time_expansion == "year":
            return [{"year": f"{year:04d}"} for year in cls._expand_years(start_date, end_date)]
        if time_expansion in {"none", "static", "single"}:
            return [{}]
        raise ValueError(f"Unsupported time_expansion: {time_expansion}")

    @staticmethod
    def _expand_years(start_date: str, end_date: str) -> list[int]:
        start_year = int(start_date.split("-")[0])
        end_year = int(end_date.split("-")[0])
        if end_year < start_year:
            raise ValueError("end_date must not be earlier than start_date")
        return list(range(start_year, end_year + 1))

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
