from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str

try:
    import paramiko
except ImportError:
    paramiko = None


class GLEAMSFTPDownloader(BaseDownloader):
    """
    GLEAM downloader using SFTP.

    Current implementation
    ----------------------
    - Intended for registered GLEAM server access
    - Requires host, port, username, password, and remote base directory
    - Assumes NetCDF files on remote server
    - Uses SFTP because the official GLEAM FAQ specifies SFTP access on port 2225

    Notes
    -----
    1. The official website requires registration before download access is granted.
    2. Login details and server access information are sent by email after registration.
    3. Because the exact remote directory layout is not public on the website, this
       downloader keeps the remote path configurable rather than hard-coding it.
    """

    provider_name = "GLEAM"
    product_variant = "monthly_netcdf_sftp"

    DEFAULT_PORT = 2225

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = DEFAULT_PORT,
        username: Optional[str] = None,
        password: Optional[str] = None,
        remote_base_dir: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 2,
        sleep_seconds: float = 2.0,
        file_limit: Optional[int] = 3,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.remote_base_dir = remote_base_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.file_limit = file_limit

    def download(self, request: DownloadRequest) -> list[Path]:
        if request.source_key.lower() != "gleam_monthly_nc":
            raise ValueError(f"Unsupported source_key for GLEAMSFTPDownloader: {request.source_key}")

        if not request.variables:
            raise ValueError("No variables were requested.")

        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")

        if request.start_date is None or request.end_date is None:
            raise ValueError("start_date and end_date must be provided.")

        if paramiko is None:
            raise ImportError("paramiko is not installed. Please install it with: pip install paramiko")

        if not self.host or not self.username or not self.password or not self.remote_base_dir:
            raise ValueError(
                "GLEAMSFTPDownloader requires host, username, password, and remote_base_dir. "
                "Fill these using the login details sent by the GLEAM server after registration."
            )

        outdir = Path(request.target_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        request_id = make_request_id(request)

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
                "sftp_host": self.host,
                "sftp_port": self.port,
                "remote_base_dir": self.remote_base_dir,
                "file_limit": self.file_limit,
            },
        )

        downloaded_files: list[Path] = []

        transport = None
        sftp = None
        try:
            recorder.log_event(
                "INFO",
                f"sftp_connect_started host={self.host} port={self.port}",
            )

            transport = paramiko.Transport((self.host, self.port))
            transport.connect(username=self.username, password=self.password)
            sftp = paramiko.SFTPClient.from_transport(transport)

            recorder.log_event(
                "INFO",
                f"sftp_connect_success host={self.host} port={self.port}",
            )

            target_files = self._list_remote_netcdf_files(
                sftp=sftp,
                remote_dir=self.remote_base_dir,
                variable_keys=request.variables,
            )

            if self.file_limit is not None:
                target_files = target_files[: self.file_limit]

            for remote_file in target_files:
                remote_path = f"{self.remote_base_dir.rstrip('/')}/{remote_file}"
                local_path = outdir / Path(remote_file).name

                if local_path.exists() and local_path.stat().st_size > 0:
                    downloaded_files.append(local_path)
                    recorder.record_skipped_existing(
                        url=f"sftp://{self.host}:{self.port}{remote_path}",
                        local_path=str(local_path),
                        size_bytes=local_path.stat().st_size,
                    )
                    continue

                success = False
                last_error = ""
                last_error_type = ""

                for attempt in range(1, self.max_retries + 1):
                    file_t0 = time.perf_counter()
                    file_started_at = now_str()

                    recorder.log_event(
                        "INFO",
                        f"remote_path={remote_path} attempt={attempt} download_started",
                    )

                    try:
                        sftp.get(remote_path, str(local_path))

                        file_elapsed = time.perf_counter() - file_t0
                        file_finished_at = now_str()
                        size_bytes = local_path.stat().st_size

                        downloaded_files.append(local_path)
                        recorder.record_success(
                            url=f"sftp://{self.host}:{self.port}{remote_path}",
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
                            url=f"sftp://{self.host}:{self.port}{remote_path}",
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
                    print(f"[FAILED] {remote_path} -> {last_error}")

        finally:
            if sftp is not None:
                sftp.close()
            if transport is not None:
                transport.close()
            recorder.finalize(outdir)

        return downloaded_files

    @staticmethod
    def _list_remote_netcdf_files(
        *,
        sftp,
        remote_dir: str,
        variable_keys: tuple[str, ...],
    ) -> list[str]:
        """
        Minimal remote file selection logic.

        Current strategy:
        - list all files in remote_dir
        - keep only .nc
        - optionally filter using variable-related keywords

        Because the exact public remote directory structure is not documented
        on the website, this selection is intentionally conservative.
        """
        names = sftp.listdir(remote_dir)
        nc_files = [n for n in names if n.lower().endswith(".nc")]

        variable_tokens = []
        for v in variable_keys:
            variable_tokens.append(v.lower())
            if v.lower() == "potential_evaporation":
                variable_tokens.extend(["ep", "pot", "potential"])
            if v.lower() == "evapotranspiration":
                variable_tokens.extend(["e", "evap", "et"])
            if v.lower() == "soil_moisture":
                variable_tokens.extend(["sm", "soil", "moisture"])

        filtered = []
        for name in nc_files:
            lname = name.lower()
            if any(tok in lname for tok in variable_tokens):
                filtered.append(name)

        # fallback: if no variable-based match, return all nc files
        return filtered if filtered else nc_files