from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import requests

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.catalog import DataSource
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str


class ESGFSearchDownloader(BaseDownloader):
    """
    Generic ESGF search-and-wget-link downloader for CMIP5/CMIP6.

    First-stage implementation
    --------------------------
    - query ESGF Search REST API
    - save search_results.json
    - save search_results.csv
    - generate wget_script_url.txt

    This implementation intentionally focuses on discovery and script generation,
    which is the safest and most portable first step for ESGF-based acquisition.
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
        limit: int = 50,
        search_type: Optional[str] = None,
        facets: Optional[dict] = None,
    ) -> None:
        self.source_config = source_config
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()
        self.limit = limit
        self.search_type = search_type
        self.facets = facets or {}

    def download(self, request: DownloadRequest) -> list[Path]:
        if request.source_key.lower() != self.source_config.key.lower():
            raise ValueError(
                f"ESGFSearchDownloader received mismatched source_key: {request.source_key} "
                f"!= {self.source_config.key}"
            )

        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")

        outdir = Path(request.target_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        params_cfg = self.source_config.params
        project = params_cfg["project"]
        search_node = params_cfg["default_search_node"]
        wget_node = params_cfg["default_wget_node"]
        search_type = self.search_type or params_cfg.get("default_type", "File")

        request_id = make_request_id(request)

        recorder = AcquisitionRunRecorder(
            request_id=request_id,
            source_key=request.source_key,
            provider_name=self.provider_name,
            product_variant=f"{project}_search_and_wget",
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
                "search_node": search_node,
                "wget_node": wget_node,
                "search_type": search_type,
                "limit": self.limit,
                "facets": self.facets,
            },
        )

        json_path = outdir / "search_results.json"
        csv_path = outdir / "search_results.csv"
        wget_txt_path = outdir / "wget_script_url.txt"

        # if outputs already exist, treat as reusable
        if json_path.exists() and csv_path.exists() and wget_txt_path.exists():
            for path in (json_path, csv_path, wget_txt_path):
                recorder.record_skipped_existing(
                    url=f"esgf://{project}/{path.name}",
                    local_path=str(path),
                    size_bytes=path.stat().st_size,
                )
            recorder.finalize(outdir)
            return [json_path, csv_path, wget_txt_path]

        search_params = self._build_search_params(project=project, search_type=search_type, request=request)

        search_url = f"{search_node}?{urlencode(search_params, doseq=True)}"
        wget_url = f"{wget_node}?{urlencode(search_params, doseq=True)}"

        success = False
        last_error = ""
        last_error_type = ""
        last_http_status = None
        outputs: list[Path] = []

        for attempt in range(1, self.max_retries + 1):
            file_t0 = time.perf_counter()
            file_started_at = now_str()

            recorder.log_event("INFO", f"url={search_url} attempt={attempt} search_started")

            try:
                with self.session.get(search_url, timeout=self.timeout) as r:
                    last_http_status = r.status_code
                    r.raise_for_status()
                    payload = r.json()

                self._write_json(json_path, payload)
                self._write_csv(csv_path, payload, project=project)
                wget_txt_path.write_text(wget_url, encoding="utf-8")

                file_elapsed = time.perf_counter() - file_t0
                file_finished_at = now_str()

                for path, url in (
                    (json_path, search_url),
                    (csv_path, search_url),
                    (wget_txt_path, wget_url),
                ):
                    recorder.record_success(
                        url=url,
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

                for path in (json_path, csv_path, wget_txt_path):
                    if path.exists():
                        try:
                            path.unlink()
                        except Exception:
                            pass

                recorder.record_failure(
                    url=search_url,
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
            print(f"[FAILED] {search_url} -> {last_error}")

        recorder.finalize(outdir)
        return outputs

    def _build_search_params(self, *, project: str, search_type: str, request: DownloadRequest) -> dict:
        params_cfg = self.source_config.params

        params = {
            "project": project,
            "type": search_type,
            "latest": params_cfg.get("latest", "true"),
            "replica": params_cfg.get("replica", "false"),
            "distrib": params_cfg.get("distrib", "true"),
            "limit": self.limit,
            "format": "application/solr+json",
        }

        facet_map = self.CMIP6_FACET_MAP if project.upper() == "CMIP6" else self.CMIP5_FACET_MAP

        # variable from DownloadRequest if not passed explicitly
        if request.variables:
            variable_key = facet_map["variable"]
            params[variable_key] = list(request.variables)

        # optional generic facets
        for user_key, value in self.facets.items():
            if user_key not in facet_map:
                continue
            params[facet_map[user_key]] = value

        return params

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _write_csv(self, path: Path, payload: dict, *, project: str) -> None:
        docs = payload.get("response", {}).get("docs", [])
        rows = []

        for doc in docs:
            rows.append(
                {
                    "id": doc.get("id"),
                    "dataset_id": doc.get("dataset_id"),
                    "title": doc.get("title"),
                    "project": doc.get("project"),
                    "variable": self._pick_first(doc, ["variable_id", "variable"]),
                    "model": self._pick_first(doc, ["source_id", "model"]),
                    "experiment": self._pick_first(doc, ["experiment_id", "experiment"]),
                    "member": self._pick_first(doc, ["member_id", "ensemble"]),
                    "table": self._pick_first(doc, ["table_id", "cmor_table"]),
                    "grid": self._pick_first(doc, ["grid_label"]),
                    "version": doc.get("version"),
                    "url": self._join_values(doc.get("url")),
                }
            )

        fieldnames = [
            "id",
            "dataset_id",
            "title",
            "project",
            "variable",
            "model",
            "experiment",
            "member",
            "table",
            "grid",
            "version",
            "url",
        ]

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _pick_first(doc: dict, keys: list[str]):
        for key in keys:
            if key in doc:
                value = doc[key]
                if isinstance(value, list):
                    return value[0] if value else None
                return value
        return None

    @staticmethod
    def _join_values(value):
        if isinstance(value, list):
            return " | ".join(str(v) for v in value)
        return value