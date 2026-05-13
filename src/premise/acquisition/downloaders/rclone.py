from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.catalog import DataSource
from premise.acquisition.logging_utils import AcquisitionRunRecorder


class RcloneDownloader(BaseDownloader):
    provider_name = "Rclone"

    def __init__(self, source_config: DataSource, remote_name: Optional[str] = None, remote_path: Optional[str] = None,
                 subdir: Optional[str] = None, timeout: int = 600, max_retries: int = 2,
                 sleep_seconds: float = 2.0, rclone_binary: str = 'rclone', dry_run: bool = False) -> None:
        self.source_config = source_config
        self.remote_name = remote_name
        self.remote_path = remote_path
        self.subdir = subdir
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.rclone_binary = rclone_binary
        self.dry_run = dry_run

    def download(self, request: DownloadRequest) -> list[Path]:
        src=self.source_config
        if request.target_dir is None: raise ValueError('target_dir must be provided.')
        if not self.remote_name or not self.remote_path: raise ValueError('RcloneDownloader requires remote_name and remote_path.')
        if shutil.which(self.rclone_binary) is None: raise FileNotFoundError(f"Could not find rclone executable '{self.rclone_binary}'.")
        outdir=Path(request.target_dir); outdir.mkdir(parents=True, exist_ok=True)
        request_id=make_request_id(request)
        default_subdir = src.params.get('default_subdir')
        subdir = self.subdir or default_subdir
        recorder=AcquisitionRunRecorder(request_id=request_id, source_key=request.source_key, provider_name=src.provider, product_variant=src.title, request_dict={"variables": list(request.variables), "start_date": request.start_date, "end_date": request.end_date, "bbox": request.bbox, "target_dir": request.target_dir, "notes": request.notes, "frequency": request.frequency, "format_preference": request.format_preference}, extra_metadata={"remote_name": self.remote_name, "remote_path": self.remote_path, "subdir": subdir, "rclone_binary": self.rclone_binary, "dry_run": self.dry_run})
        before={p.name:p for p in outdir.glob('*') if p.is_file()}
        remote=f"{self.remote_name}:{self.remote_path}".rstrip('/')
        if subdir: remote=f"{remote}/{subdir.strip('/')}"
        cmd=[self.rclone_binary, 'copy', '-v', '--drive-shared-with-me', remote, str(outdir)]
        if self.dry_run: cmd.append('--dry-run')
        recorder.log_event('INFO', f"rclone_command_started cmd={' '.join(cmd)}")
        success=False; last_error=''; last_error_type=''
        for attempt in range(1, self.max_retries+1):
            tt=time.perf_counter();
            try:
                result=subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, check=False)
                elapsed=time.perf_counter()-tt
                if result.returncode != 0:
                    last_error=(result.stderr or result.stdout or '').strip(); last_error_type='RcloneError'
                    recorder.record_failure(url=remote, local_path=str(outdir), attempt=attempt, download_started_at=None, download_finished_at=None, elapsed_seconds=elapsed, http_status=None, error_type=last_error_type, error_message=last_error, file_exists_after_download=outdir.exists())
                    if attempt < self.max_retries: time.sleep(self.sleep_seconds)
                    continue
                recorder.log_event('INFO', f"rclone_command_success attempt={attempt} elapsed_seconds={round(elapsed, 3)}")
                success=True; break
            except Exception as e:
                elapsed=time.perf_counter()-tt; last_error=str(e); last_error_type=type(e).__name__
                recorder.record_failure(url=remote, local_path=str(outdir), attempt=attempt, download_started_at=None, download_finished_at=None, elapsed_seconds=elapsed, http_status=None, error_type=last_error_type, error_message=last_error, file_exists_after_download=outdir.exists())
                if attempt < self.max_retries: time.sleep(self.sleep_seconds)
        if not success:
            print(f"[FAILED] Rclone request -> {last_error}"); recorder.finalize(outdir); return []
        after={p.name:p for p in outdir.glob('*') if p.is_file()}
        files=[]
        for name, path in after.items():
            size_bytes=path.stat().st_size
            if name in before:
                recorder.record_skipped_existing(url=f"rclone://{remote}/{name}", local_path=str(path), size_bytes=size_bytes)
            else:
                recorder.record_success(url=f"rclone://{remote}/{name}", local_path=str(path), attempt=1, download_started_at=None, download_finished_at=None, elapsed_seconds=0.0, size_bytes=size_bytes, http_status=None, file_exists_after_download=path.exists())
            files.append(path)
        recorder.finalize(outdir)
        return sorted(files)
