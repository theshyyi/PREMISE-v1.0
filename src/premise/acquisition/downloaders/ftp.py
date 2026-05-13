from __future__ import annotations

import ftplib
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.catalog import DataSource
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str


class FTPDownloader(BaseDownloader):
    provider_name = "FTP"

    def __init__(
        self,
        source_config: DataSource,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        sleep_seconds: float = 1.0,
        chunk_size: int = 1024 * 1024,
        file_limit: Optional[int] = None,
        use_tls: Optional[bool] = None,
        passive: bool = True,
    ) -> None:
        self.source_config = source_config
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.chunk_size = chunk_size
        self.file_limit = file_limit
        self.use_tls = use_tls
        self.passive = passive

    def download(self, request: DownloadRequest) -> list[Path]:
        src = self.source_config
        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")
        if request.start_date is None or request.end_date is None:
            raise ValueError("start_date and end_date must be provided.")

        params = src.params
        host = self.host or params.get("host")
        port = self.port or params.get("default_port", 21)
        use_tls = self.use_tls if self.use_tls is not None else bool(params.get("use_tls", False))
        username = self.username if self.username is not None else params.get("default_username", "anonymous")
        password = self.password if self.password is not None else params.get("default_password", "anonymous@")
        time_expansion = params.get("time_expansion", "date")
        file_limit = self.file_limit if self.file_limit is not None else params.get("file_limit_default")

        if not host:
            raise ValueError("FTPDownloader requires host in downloader kwargs or source params.")

        outdir = Path(request.target_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        request_id = make_request_id(request)

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
                "ftp_host": host,
                "ftp_port": port,
                "use_tls": use_tls,
                "time_expansion": time_expansion,
            },
        )

        ftp = self._connect(host=host, port=port, username=username, password=password, use_tls=use_tls)
        ftp.set_pasv(self.passive)
        recorder.log_event("INFO", f"ftp_connect_success host={host} port={port} use_tls={use_tls}")

        discovered: list[tuple[str, str]] = []
        seen_remote: set[str] = set()
        try:
            for ctx in self._expand_contexts(request.start_date, request.end_date, time_expansion=time_expansion):
                base_dir = (params.get("remote_base_dir", "") or "").rstrip("/")
                directory_pattern = params.get("directory_pattern", "") or ""
                remote_dir = f"{base_dir}/{directory_pattern.format(**ctx)}".replace("//", "/").rstrip("/")
                if not remote_dir:
                    remote_dir = "/"

                filename_pattern = params.get("filename_pattern")
                if filename_pattern:
                    name = filename_pattern.format(**ctx)
                    remote_path = self._join_remote(remote_dir, name)
                    if remote_path not in seen_remote:
                        seen_remote.add(remote_path)
                        discovered.append((remote_path, name))
                    continue

                recorder.log_event("INFO", f"ftp_listing_started remote_dir={remote_dir}")
                try:
                    names = ftp.nlst(remote_dir)
                except Exception as e:
                    recorder.record_failure(
                        url=f"ftp://{host}:{port}{remote_dir}",
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
                    print(f"[FAILED] listing ftp://{host}:{port}{remote_dir} -> {e}")
                    continue

                filtered_names = self._filter_names(names, request=request, params=params, ctx=ctx)
                for remote_path in filtered_names:
                    clean_name = remote_path.split("/")[-1]
                    if remote_path not in seen_remote:
                        seen_remote.add(remote_path)
                        discovered.append((remote_path, clean_name))

            if file_limit is not None:
                discovered = discovered[:file_limit]

            downloaded: list[Path] = []
            for remote_path, name in discovered:
                local_path = outdir / Path(name).name
                url = f"ftp://{host}:{port}{remote_path}"
                if local_path.exists() and local_path.stat().st_size > 0:
                    downloaded.append(local_path)
                    recorder.record_skipped_existing(url=url, local_path=str(local_path), size_bytes=local_path.stat().st_size)
                    continue

                success = False
                last_error = ""
                last_error_type = ""
                for attempt in range(1, self.max_retries + 1):
                    t0 = time.perf_counter()
                    started = now_str()
                    recorder.log_event("INFO", f"remote_path={remote_path} attempt={attempt} download_started")
                    try:
                        with open(local_path, "wb") as f:
                            ftp.retrbinary(f"RETR {remote_path}", f.write, blocksize=self.chunk_size)
                        elapsed = time.perf_counter() - t0
                        finished = now_str()
                        size_bytes = local_path.stat().st_size
                        downloaded.append(local_path)
                        recorder.record_success(
                            url=url,
                            local_path=str(local_path),
                            attempt=attempt,
                            download_started_at=started,
                            download_finished_at=finished,
                            elapsed_seconds=elapsed,
                            size_bytes=size_bytes,
                            http_status=None,
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
                            url=url,
                            local_path=str(local_path),
                            attempt=attempt,
                            download_started_at=started,
                            download_finished_at=finished,
                            elapsed_seconds=elapsed,
                            http_status=None,
                            error_type=last_error_type,
                            error_message=last_error,
                            file_exists_after_download=local_path.exists(),
                        )
                        if attempt < self.max_retries:
                            time.sleep(self.sleep_seconds)
                if not success:
                    print(f"[FAILED] {url} -> {last_error}")
        finally:
            try:
                ftp.quit()
            except Exception:
                try:
                    ftp.close()
                except Exception:
                    pass
            recorder.finalize(outdir)

        return downloaded

    def _filter_names(self, names: list[str], *, request: DownloadRequest, params: dict, ctx: dict) -> list[str]:
        file_regex = params.get("file_regex")
        regex = re.compile(file_regex, re.IGNORECASE) if file_regex else None

        contains_cfg = params.get("contains", [])
        contains = [c.lower() for c in contains_cfg]

        contains_template_cfg = params.get("contains_template", [])
        if isinstance(contains_template_cfg, str):
            contains_template_cfg = [contains_template_cfg]
        formatted_contains = [token.format(**ctx).lower() for token in contains_template_cfg]

        filtered: list[str] = []
        seen: set[str] = set()
        for raw in names:
            remote_path = raw.strip()
            name = remote_path.split("/")[-1]
            lower_name = name.lower()
            if not name or name in {".", ".."}:
                continue
            if regex and not regex.search(name):
                continue
            if contains and not all(token in lower_name for token in contains):
                continue
            if formatted_contains and not all(token in lower_name for token in formatted_contains):
                continue
            if not self._filename_in_requested_window(name=name, start_date=request.start_date, end_date=request.end_date, params=params):
                continue
            if remote_path not in seen:
                filtered.append(remote_path)
                seen.add(remote_path)
        return filtered

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

    @staticmethod
    def _join_remote(remote_dir: str, name: str) -> str:
        if remote_dir in {"", "/"}:
            return "/" + name.lstrip("/")
        return remote_dir.rstrip("/") + "/" + name.lstrip("/")

    @staticmethod
    def _connect(*, host: str, port: int, username: str, password: str, use_tls: bool):
        if use_tls:
            ftp = ftplib.FTP_TLS()
            ftp.connect(host=host, port=port, timeout=120)
            ftp.login(user=username, passwd=password)
            ftp.prot_p()
            return ftp
        ftp = ftplib.FTP()
        ftp.connect(host=host, port=port, timeout=120)
        ftp.login(user=username, passwd=password)
        return ftp
