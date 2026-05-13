from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.catalog import DataSource
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str

try:
    import paramiko
except ImportError:
    paramiko = None


class SFTPDownloader(BaseDownloader):
    provider_name = "SFTP"

    def __init__(self, source_config: DataSource, host: Optional[str] = None, port: Optional[int] = None,
                 username: Optional[str] = None, password: Optional[str] = None, remote_base_dir: Optional[str] = None,
                 timeout: int = 120, max_retries: int = 2, sleep_seconds: float = 2.0, file_limit: Optional[int] = None) -> None:
        self.source_config = source_config
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
        src=self.source_config
        if paramiko is None: raise ImportError("paramiko is not installed. Please install it with: pip install paramiko")
        if request.target_dir is None: raise ValueError("target_dir must be provided.")
        if not self.host or not self.username or not self.password:
            raise ValueError("SFTPDownloader requires host, username, and password.")
        outdir=Path(request.target_dir); outdir.mkdir(parents=True, exist_ok=True)
        request_id=make_request_id(request)
        params=src.params
        port = self.port or params.get('default_port', 22)
        remote_base_dir = self.remote_base_dir
        if remote_base_dir is None:
            mapping = params.get('variable_directory_map', {})
            remote_base_dir = mapping.get(request.variables[0]) if len(request.variables)==1 else None
            if remote_base_dir is None:
                raise ValueError("remote_base_dir must be provided for this source/variable combination.")
        recorder=AcquisitionRunRecorder(request_id=request_id, source_key=request.source_key, provider_name=src.provider, product_variant=src.title, request_dict={"variables": list(request.variables), "start_date": request.start_date, "end_date": request.end_date, "bbox": request.bbox, "target_dir": request.target_dir, "notes": request.notes, "frequency": request.frequency, "format_preference": request.format_preference}, extra_metadata={"sftp_host": self.host, "sftp_port": port, "remote_base_dir": remote_base_dir, "file_limit": self.file_limit if self.file_limit is not None else params.get('file_limit_default')})
        transport=None; sftp=None; downloaded=[]
        try:
            recorder.log_event('INFO', f'sftp_connect_started host={self.host} port={port}')
            transport = paramiko.Transport((self.host, port)); transport.connect(username=self.username, password=self.password)
            sftp = paramiko.SFTPClient.from_transport(transport)
            recorder.log_event('INFO', f'sftp_connect_success host={self.host} port={port}')
            names = [n for n in sftp.listdir(remote_base_dir) if n.lower().endswith('.nc')]
            limit = self.file_limit if self.file_limit is not None else params.get('file_limit_default')
            if limit is not None: names = names[:limit]
            for name in names:
                remote_path = f"{remote_base_dir.rstrip('/')}/{name}"
                local_path = outdir / Path(name).name
                if local_path.exists() and local_path.stat().st_size>0:
                    downloaded.append(local_path); recorder.record_skipped_existing(url=f"sftp://{self.host}:{port}{remote_path}", local_path=str(local_path), size_bytes=local_path.stat().st_size); continue
                success=False; last_error=''; last_error_type=''
                for attempt in range(1, self.max_retries+1):
                    tt=time.perf_counter(); started=now_str(); recorder.log_event('INFO', f'remote_path={remote_path} attempt={attempt} download_started')
                    try:
                        sftp.get(remote_path, str(local_path))
                        elapsed=time.perf_counter()-tt; finished=now_str(); size_bytes=local_path.stat().st_size
                        downloaded.append(local_path)
                        recorder.record_success(url=f"sftp://{self.host}:{port}{remote_path}", local_path=str(local_path), attempt=attempt, download_started_at=started, download_finished_at=finished, elapsed_seconds=elapsed, size_bytes=size_bytes, http_status=None, file_exists_after_download=local_path.exists())
                        success=True; break
                    except Exception as e:
                        elapsed=time.perf_counter()-tt; finished=now_str(); last_error=str(e); last_error_type=type(e).__name__
                        if local_path.exists():
                            try: local_path.unlink()
                            except Exception: pass
                        recorder.record_failure(url=f"sftp://{self.host}:{port}{remote_path}", local_path=str(local_path), attempt=attempt, download_started_at=started, download_finished_at=finished, elapsed_seconds=elapsed, http_status=None, error_type=last_error_type, error_message=last_error, file_exists_after_download=local_path.exists())
                        if attempt < self.max_retries: time.sleep(self.sleep_seconds)
                if not success: print(f"[FAILED] {remote_path} -> {last_error}")
        finally:
            if sftp is not None: sftp.close()
            if transport is not None: transport.close()
            recorder.finalize(outdir)
        return downloaded
