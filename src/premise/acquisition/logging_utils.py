from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def bytes_to_mb(size_bytes: int) -> float:
    return size_bytes / (1024 * 1024)


class AcquisitionRunRecorder:
    """
    Shared task/file-level recorder for acquisition workflows.
    All provider-specific downloaders can reuse this class.
    """

    def __init__(
        self,
        *,
        request_id: str,
        source_key: str,
        provider_name: str,
        product_variant: str,
        request_dict: dict[str, Any],
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.request_id = request_id
        self.source_key = source_key
        self.provider_name = provider_name
        self.product_variant = product_variant
        self.request_dict = request_dict
        self.extra_metadata = extra_metadata or {}

        self.requested_at = now_str()
        self.started_at = now_str()
        self.task_t0 = time.perf_counter()

        self.records: list[dict[str, Any]] = []
        self.event_lines: list[str] = []
        self.log_event("INFO", f"request_id={self.request_id} source={self.source_key} task_started")

    def log_event(self, level: str, message: str) -> None:
        self.event_lines.append(f"[{now_str()}] {level:<7} {message}")

    def record_skipped_existing(
        self,
        *,
        url: str,
        local_path: str,
        size_bytes: int,
    ) -> None:
        self.records.append(
            {
                "url": url,
                "local_path": local_path,
                "status": "skipped_existing",
                "attempt": 0,
                "download_started_at": None,
                "download_finished_at": None,
                "elapsed_seconds": 0.0,
                "size_bytes": size_bytes,
                "size_mb": round(bytes_to_mb(size_bytes), 3),
                "throughput_MBps": None,
                "file_exists_after_download": True,
                "http_status": None,
                "error_type": None,
                "error_message": "",
            }
        )
        self.log_event(
            "INFO",
            f"url={url} status=skipped_existing size_bytes={size_bytes}",
        )

    def record_success(
        self,
        *,
        url: str,
        local_path: str,
        attempt: int,
        download_started_at: str,
        download_finished_at: str,
        elapsed_seconds: float,
        size_bytes: int,
        http_status: int | None,
        file_exists_after_download: bool,
    ) -> None:
        size_mb = bytes_to_mb(size_bytes)
        throughput = size_mb / elapsed_seconds if elapsed_seconds > 0 else None

        self.records.append(
            {
                "url": url,
                "local_path": local_path,
                "status": "success",
                "attempt": attempt,
                "download_started_at": download_started_at,
                "download_finished_at": download_finished_at,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "size_bytes": size_bytes,
                "size_mb": round(size_mb, 3),
                "throughput_MBps": round(throughput, 3) if throughput is not None else None,
                "file_exists_after_download": file_exists_after_download,
                "http_status": http_status,
                "error_type": None,
                "error_message": "",
            }
        )

        self.log_event(
            "INFO",
            f"url={url} status=success attempt={attempt} size_bytes={size_bytes} elapsed_seconds={round(elapsed_seconds, 3)}",
        )

    def record_failure(
        self,
        *,
        url: str,
        local_path: str,
        attempt: int,
        download_started_at: str | None,
        download_finished_at: str | None,
        elapsed_seconds: float | None,
        http_status: int | None,
        error_type: str,
        error_message: str,
        file_exists_after_download: bool,
    ) -> None:
        self.records.append(
            {
                "url": url,
                "local_path": local_path,
                "status": "failed",
                "attempt": attempt,
                "download_started_at": download_started_at,
                "download_finished_at": download_finished_at,
                "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
                "size_bytes": 0,
                "size_mb": 0.0,
                "throughput_MBps": None,
                "file_exists_after_download": file_exists_after_download,
                "http_status": http_status,
                "error_type": error_type,
                "error_message": error_message,
            }
        )

        self.log_event(
            "WARNING",
            f"url={url} status=failed attempt={attempt} elapsed_seconds={round(elapsed_seconds, 3) if elapsed_seconds is not None else None} error_type={error_type} error={error_message}",
        )

    def finalize(self, outdir: str | Path) -> dict[str, Any]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        finished_at = now_str()
        total_elapsed = time.perf_counter() - self.task_t0

        requested_count = len(self.records)
        success_count = sum(1 for r in self.records if r["status"] == "success")
        failed_count = sum(1 for r in self.records if r["status"] == "failed")
        skipped_count = sum(1 for r in self.records if r["status"] == "skipped_existing")

        total_downloaded_bytes = sum(r.get("size_bytes", 0) for r in self.records if r["status"] == "success")
        total_downloaded_mb = bytes_to_mb(total_downloaded_bytes)

        successful_elapsed = [
            r["elapsed_seconds"]
            for r in self.records
            if r["status"] == "success" and r["elapsed_seconds"] is not None
        ]
        successful_throughput = [
            r["throughput_MBps"]
            for r in self.records
            if r["status"] == "success" and r["throughput_MBps"] is not None
        ]

        self.log_event(
            "INFO",
            f"request_id={self.request_id} task_finished total_elapsed_seconds={round(total_elapsed, 3)}",
        )

        manifest = {
            "request_id": self.request_id,
            "source_key": self.source_key,
            "provider_name": self.provider_name,
            "product_variant": self.product_variant,
            **self.request_dict,
            **self.extra_metadata,
            "requested_at": self.requested_at,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "total_elapsed_seconds": round(total_elapsed, 3),
            "requested_file_count": requested_count,
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_existing_count": skipped_count,
            "success_rate": round(success_count / requested_count, 3) if requested_count > 0 else None,
            "total_downloaded_bytes": total_downloaded_bytes,
            "total_downloaded_mb": round(total_downloaded_mb, 3),
            "mean_seconds_per_file": round(total_elapsed / requested_count, 3) if requested_count > 0 else None,
            "mean_successful_file_seconds": round(mean(successful_elapsed), 3) if successful_elapsed else None,
            "mean_throughput_MBps": round(mean(successful_throughput), 3) if successful_throughput else None,
            "created_at": now_str(),
            "records": self.records,
        }

        self._write_manifest_json(outdir, manifest)
        self._write_summary_csv(outdir, manifest)
        self._write_file_records_csv(outdir, self.request_id, self.records)
        self._write_event_log(outdir, self.event_lines)

        return manifest

    @staticmethod
    def _write_manifest_json(outdir: Path, manifest: dict[str, Any]) -> None:
        with open(outdir / "acquisition_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _write_summary_csv(outdir: Path, manifest: dict[str, Any]) -> None:
        summary_path = outdir / "acquisition_summary.csv"
        fieldnames = [
            "request_id",
            "source_key",
            "provider_name",
            "product_variant",
            "variables",
            "start_date",
            "end_date",
            "requested_file_count",
            "success_count",
            "failed_count",
            "skipped_existing_count",
            "success_rate",
            "total_downloaded_bytes",
            "total_downloaded_mb",
            "total_elapsed_seconds",
            "mean_seconds_per_file",
            "mean_successful_file_seconds",
            "mean_throughput_MBps",
            "target_dir",
            "created_at",
        ]
        row = {
            "request_id": manifest["request_id"],
            "source_key": manifest["source_key"],
            "provider_name": manifest["provider_name"],
            "product_variant": manifest["product_variant"],
            "variables": ";".join(manifest.get("variables", [])),
            "start_date": manifest.get("start_date"),
            "end_date": manifest.get("end_date"),
            "requested_file_count": manifest["requested_file_count"],
            "success_count": manifest["success_count"],
            "failed_count": manifest["failed_count"],
            "skipped_existing_count": manifest["skipped_existing_count"],
            "success_rate": manifest["success_rate"],
            "total_downloaded_bytes": manifest["total_downloaded_bytes"],
            "total_downloaded_mb": manifest["total_downloaded_mb"],
            "total_elapsed_seconds": manifest["total_elapsed_seconds"],
            "mean_seconds_per_file": manifest["mean_seconds_per_file"],
            "mean_successful_file_seconds": manifest["mean_successful_file_seconds"],
            "mean_throughput_MBps": manifest["mean_throughput_MBps"],
            "target_dir": manifest.get("target_dir"),
            "created_at": manifest["created_at"],
        }

        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _write_file_records_csv(outdir: Path, request_id: str, records: list[dict[str, Any]]) -> None:
        record_path = outdir / "acquisition_file_records.csv"
        fieldnames = [
            "request_id",
            "url",
            "local_path",
            "status",
            "attempt",
            "download_started_at",
            "download_finished_at",
            "elapsed_seconds",
            "size_bytes",
            "size_mb",
            "throughput_MBps",
            "file_exists_after_download",
            "http_status",
            "error_type",
            "error_message",
        ]

        with open(record_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                row = {"request_id": request_id}
                row.update(r)
                writer.writerow(row)

    @staticmethod
    def _write_event_log(outdir: Path, event_lines: list[str]) -> None:
        log_path = outdir / "acquisition_events.log"
        with open(log_path, "w", encoding="utf-8") as f:
            for line in event_lines:
                f.write(line + "\n")