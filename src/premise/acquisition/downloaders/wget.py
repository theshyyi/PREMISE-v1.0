from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import requests

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.catalog import DataSource
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str


class ESGFWgetDownloader(BaseDownloader):
    """
    ESGF wget-script downloader and optional executor.

    Workflow
    --------
    1. Build wget-script URL from ESGF facet constraints
    2. Download the wget shell script to local disk
    3. Optionally execute the script with a user-supplied shell

    Notes
    -----
    - On Windows, actual execution usually requires WSL, Git Bash, or another bash-compatible shell.
    - If execute_script=False, this downloader still provides a fully resolved wget script ready for manual use.
    """

    provider_name = "ESGF"

    CMIP6_FACET_MAP = {
        "model": "source_id",
        "experiment": "experiment_id",
        "member": "member_id",
        "table": "table_id",
        "variable": "variable_id",
        "grid": "grid_label",
        "frequency": "frequency",
        "realm": "realm",
    }

    CMIP5_FACET_MAP = {
        "model": "model",
        "experiment": "experiment",
        "member": "ensemble",
        "table": "cmor_table",
        "variable": "variable",
        "frequency": "time_frequency",
        "realm": "realm",
    }

    def __init__(
        self,
        source_config: DataSource,
        timeout: int = 120,
        max_retries: int = 3,
        sleep_seconds: float = 1.0,
        session: Optional[requests.Session] = None,
        limit: int = 100,
        facets: Optional[dict] = None,
        execute_script: bool = False,
        bash_executable: Optional[str] = None,
        script_name: Optional[str] = None,
    ) -> None:
        self.source_config = source_config
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()
        self.limit = limit
        self.facets = facets or {}
        self.execute_script = execute_script
        self.bash_executable = bash_executable
        self.script_name = script_name

    def download(self, request: DownloadRequest) -> list[Path]:
        if request.source_key.lower() != self.source_config.key.lower():
            raise ValueError(
                f"ESGFWgetDownloader received mismatched source_key: {request.source_key} "
                f"!= {self.source_config.key}"
            )

        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")

        outdir = Path(request.target_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        params_cfg = self.source_config.params
        project = params_cfg["project"]
        wget_node = params_cfg["default_wget_node"]

        request_id = make_request_id(request)

        recorder = AcquisitionRunRecorder(
            request_id=request_id,
            source_key=request.source_key,
            provider_name=self.provider_name,
            product_variant=f"{project}_wget_script",
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
                "project": project,
                "wget_node": wget_node,
                "limit": self.limit,
                "facets": self.facets,
                "execute_script": self.execute_script,
                "bash_executable": self.bash_executable,
            },
        )

        wget_params = self._build_wget_params(project=project, request=request)
        wget_url = f"{wget_node}?{urlencode(wget_params, doseq=True)}"

        script_path = outdir / (
            self.script_name
            or f"{request.source_key}_{'-'.join(request.variables) if request.variables else 'files'}.sh"
        )
        wget_url_path = outdir / "wget_script_url.txt"
        run_log_path = outdir / "wget_execution.log"

        outputs: list[Path] = []

        if script_path.exists() and wget_url_path.exists():
            for path in (script_path, wget_url_path):
                recorder.record_skipped_existing(
                    url=wget_url,
                    local_path=str(path),
                    size_bytes=path.stat().st_size,
                )
                outputs.append(path)

            if run_log_path.exists():
                outputs.append(run_log_path)

            recorder.finalize(outdir)
            return outputs

        success = False
        last_error = ""
        last_error_type = ""
        last_http_status = None

        # step 1: download wget shell script
        for attempt in range(1, self.max_retries + 1):
            file_t0 = time.perf_counter()
            file_started_at = now_str()

            recorder.log_event("INFO", f"url={wget_url} attempt={attempt} wget_script_download_started")

            try:
                with self.session.get(wget_url, timeout=self.timeout) as r:
                    last_http_status = r.status_code
                    r.raise_for_status()
                    script_path.write_text(r.text, encoding="utf-8")

                wget_url_path.write_text(wget_url, encoding="utf-8")

                file_elapsed = time.perf_counter() - file_t0
                file_finished_at = now_str()

                for path in (script_path, wget_url_path):
                    recorder.record_success(
                        url=wget_url,
                        local_path=str(path),
                        attempt=attempt,
                        download_started_at=file_started_at,
                        download_finished_at=file_finished_at,
                        elapsed_seconds=file_elapsed,
                        size_bytes=path.stat().st_size,
                        http_status=last_http_status,
                        file_exists_after_download=path.exists(),
                    )
                    outputs.append(path)

                success = True
                break

            except Exception as e:
                file_elapsed = time.perf_counter() - file_t0
                file_finished_at = now_str()
                last_error = str(e)
                last_error_type = type(e).__name__

                for path in (script_path, wget_url_path):
                    if path.exists():
                        try:
                            path.unlink()
                        except Exception:
                            pass

                recorder.record_failure(
                    url=wget_url,
                    local_path=str(outdir),
                    attempt=attempt,
                    download_started_at=file_started_at,
                    download_finished_at=file_finished_at,
                    elapsed_seconds=file_elapsed,
                    http_status=last_http_status,
                    error_type=last_error_type,
                    error_message=last_error,
                    file_exists_after_download=outdir.exists(),
                )

                if attempt < self.max_retries:
                    time.sleep(self.sleep_seconds)

        if not success:
            print(f"[FAILED] {wget_url} -> {last_error}")
            recorder.finalize(outdir)
            return outputs

        # step 2: optional execution
        if self.execute_script:
            if not self.bash_executable:
                raise ValueError(
                    "execute_script=True requires bash_executable, "
                    "for example a WSL bash or Git Bash path."
                )

            exec_t0 = time.perf_counter()
            exec_started_at = now_str()
            recorder.log_event("INFO", f"wget_script_execution_started script={script_path}")

            try:
                result = subprocess.run(
                    [self.bash_executable, str(script_path)],
                    cwd=str(outdir),
                    capture_output=True,
                    text=True,
                    timeout=None,
                    check=False,
                )

                run_log_path.write_text(
                    (result.stdout or "") + "\n\n[STDERR]\n" + (result.stderr or ""),
                    encoding="utf-8",
                )

                exec_elapsed = time.perf_counter() - exec_t0
                exec_finished_at = now_str()

                if result.returncode != 0:
                    recorder.record_failure(
                        url=str(script_path),
                        local_path=str(run_log_path),
                        attempt=1,
                        download_started_at=exec_started_at,
                        download_finished_at=exec_finished_at,
                        elapsed_seconds=exec_elapsed,
                        http_status=None,
                        error_type="WgetScriptExecutionError",
                        error_message=f"wget script exited with return code {result.returncode}",
                        file_exists_after_download=run_log_path.exists(),
                    )
                else:
                    recorder.record_success(
                        url=str(script_path),
                        local_path=str(run_log_path),
                        attempt=1,
                        download_started_at=exec_started_at,
                        download_finished_at=exec_finished_at,
                        elapsed_seconds=exec_elapsed,
                        size_bytes=run_log_path.stat().st_size if run_log_path.exists() else 0,
                        http_status=None,
                        file_exists_after_download=run_log_path.exists(),
                    )

                outputs.append(run_log_path)

            except Exception as e:
                exec_elapsed = time.perf_counter() - exec_t0
                exec_finished_at = now_str()

                run_log_path.write_text(str(e), encoding="utf-8")

                recorder.record_failure(
                    url=str(script_path),
                    local_path=str(run_log_path),
                    attempt=1,
                    download_started_at=exec_started_at,
                    download_finished_at=exec_finished_at,
                    elapsed_seconds=exec_elapsed,
                    http_status=None,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    file_exists_after_download=run_log_path.exists(),
                )
                outputs.append(run_log_path)

        recorder.finalize(outdir)
        return outputs

    def _build_wget_params(self, *, project: str, request: DownloadRequest) -> dict:
        params_cfg = self.source_config.params

        params = {
            "project": project,
            "latest": params_cfg.get("latest", "true"),
            "replica": params_cfg.get("replica", "false"),
            "distrib": params_cfg.get("distrib", "true"),
            "limit": self.limit,
        }

        facet_map = self.CMIP6_FACET_MAP if project.upper() == "CMIP6" else self.CMIP5_FACET_MAP

        if request.variables:
            variable_key = facet_map["variable"]
            params[variable_key] = list(request.variables)

        for user_key, value in self.facets.items():
            if user_key not in facet_map:
                continue
            params[facet_map[user_key]] = value

        return params