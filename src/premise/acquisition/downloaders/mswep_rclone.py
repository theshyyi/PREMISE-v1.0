from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str


class MSWEPRcloneDownloader(BaseDownloader):
    """
    MSWEP downloader using rclone.

    Official access notes
    ---------------------
    - Noncommercial users currently download MSWEP via rclone from a shared Google Drive.
    - Access is granted after submitting a request and receiving email instructions.
    - This downloader assumes the user has already:
        1. installed rclone
        2. configured a Google Drive remote
        3. confirmed the shared MSWEP folder is visible to rclone

    Typical workflow
    ----------------
    - remote_name: e.g. "GoogleDrive"
    - remote_path: path shown by `rclone lsd --drive-shared-with-me GoogleDrive:`
    - subdir selection: e.g. "Daily", "Monthly", "3hourly", depending on release structure
    """

    provider_name = "MSWEP"
    product_variant = "rclone_shared_drive"

    def __init__(
        self,
        remote_name: Optional[str] = None,
        remote_path: Optional[str] = None,
        subdir: Optional[str] = None,
        timeout: int = 600,
        max_retries: int = 2,
        sleep_seconds: float = 2.0,
        rclone_binary: str = "rclone",
        dry_run: bool = False,
    ) -> None:
        self.remote_name = remote_name
        self.remote_path = remote_path
        self.subdir = subdir
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.rclone_binary = rclone_binary
        self.dry_run = dry_run

    def download(self, request: DownloadRequest) -> list[Path]:
        if request.source_key.lower() != "mswep_rclone":
            raise ValueError(f"Unsupported source_key for MSWEPRcloneDownloader: {request.source_key}")

        if not request.variables:
            raise ValueError("No variables were requested.")

        vars_lower = {v.lower() for v in request.variables}
        if "precipitation" not in vars_lower:
            raise ValueError("MSWEPRcloneDownloader currently supports only precipitation.")

        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")

        if not self.remote_name or not self.remote_path:
            raise ValueError(
                "MSWEPRcloneDownloader requires remote_name and remote_path. "
                "Use the shared Google Drive information provided in the MSWEP access email."
            )

        if shutil.which(self.rclone_binary) is None:
            raise FileNotFoundError(
                f"Could not find rclone executable '{self.rclone_binary}'. "
                "Please install rclone and ensure it is available on PATH."
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
                "remote_name": self.remote_name,
                "remote_path": self.remote_path,
                "subdir": self.subdir,
                "rclone_binary": self.rclone_binary,
                "dry_run": self.dry_run,
            },
        )

        downloaded_files_before = {p.name: p for p in outdir.glob("*") if p.is_file()}

        remote_spec = self._build_remote_spec()
        cmd = self._build_command(remote_spec=remote_spec, local_dir=str(outdir))

        recorder.log_event("INFO", f"rclone_command_started cmd={' '.join(cmd)}")

        success = False
        last_error = ""
        last_error_type = ""

        for attempt in range(1, self.max_retries + 1):
            file_t0 = time.perf_counter()
            file_started_at = now_str()

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )

                file_elapsed = time.perf_counter() - file_t0
                file_finished_at = now_str()

                if result.returncode != 0:
                    last_error = (result.stderr or result.stdout or "").strip()
                    last_error_type = "RcloneError"

                    recorder.record_failure(
                        url=remote_spec,
                        local_path=str(outdir),
                        attempt=attempt,
                        download_started_at=file_started_at,
                        download_finished_at=file_finished_at,
                        elapsed_seconds=file_elapsed,
                        http_status=None,
                        error_type=last_error_type,
                        error_message=last_error,
                        file_exists_after_download=outdir.exists(),
                    )

                    if attempt < self.max_retries:
                        time.sleep(self.sleep_seconds)
                    continue

                success = True
                recorder.log_event(
                    "INFO",
                    f"rclone_command_success attempt={attempt} elapsed_seconds={round(file_elapsed, 3)}",
                )
                break

            except Exception as e:
                file_elapsed = time.perf_counter() - file_t0
                file_finished_at = now_str()
                last_error = str(e)
                last_error_type = type(e).__name__

                recorder.record_failure(
                    url=remote_spec,
                    local_path=str(outdir),
                    attempt=attempt,
                    download_started_at=file_started_at,
                    download_finished_at=file_finished_at,
                    elapsed_seconds=file_elapsed,
                    http_status=None,
                    error_type=last_error_type,
                    error_message=last_error,
                    file_exists_after_download=outdir.exists(),
                )

                if attempt < self.max_retries:
                    time.sleep(self.sleep_seconds)

        if not success:
            print(f"[FAILED] Rclone request -> {last_error}")
            recorder.finalize(outdir)
            return []

        downloaded_files_after = {p.name: p for p in outdir.glob("*") if p.is_file()}

        new_or_existing_files = []
        for name, path in downloaded_files_after.items():
            size_bytes = path.stat().st_size
            if name in downloaded_files_before:
                recorder.record_skipped_existing(
                    url=f"rclone://{remote_spec}/{name}",
                    local_path=str(path),
                    size_bytes=size_bytes,
                )
            else:
                recorder.record_success(
                    url=f"rclone://{remote_spec}/{name}",
                    local_path=str(path),
                    attempt=1,
                    download_started_at=None,
                    download_finished_at=now_str(),
                    elapsed_seconds=0.0,
                    size_bytes=size_bytes,
                    http_status=None,
                    file_exists_after_download=path.exists(),
                )
            new_or_existing_files.append(path)

        recorder.finalize(outdir)
        return sorted(new_or_existing_files)

    def _build_remote_spec(self) -> str:
        remote = f"{self.remote_name}:{self.remote_path}".rstrip("/")
        if self.subdir:
            remote = f"{remote}/{self.subdir.strip('/')}"
        return remote

    def _build_command(self, *, remote_spec: str, local_dir: str) -> list[str]:
        cmd = [
            self.rclone_binary,
            "copy",
            "-v",
            "--drive-shared-with-me",
            remote_spec,
            local_dir,
        ]
        if self.dry_run:
            cmd.append("--dry-run")
        return cmd