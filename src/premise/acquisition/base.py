from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DownloadRequest:
    """User-defined request for an acquisition task."""

    source_key: str
    variables: tuple[str, ...]
    start_date: str | None = None
    end_date: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    target_dir: str | None = None
    notes: str = ""
    request_id: str | None = None
    frequency: str | None = None
    format_preference: str | None = None


def make_request_id(req: DownloadRequest) -> str:
    if req.request_id:
        return req.request_id
    var_part = "-".join(req.variables) if req.variables else "unknownvar"
    start_part = req.start_date.replace("-", "") if req.start_date else "unknownstart"
    end_part = req.end_date.replace("-", "") if req.end_date else "unknownend"
    return f"{req.source_key}_{var_part}_{start_part}_{end_part}"


class BaseDownloader:
    provider_name = "generic"

    def download(self, request: DownloadRequest) -> list[Path]:
        raise NotImplementedError


def build_download_plan(requests: Iterable[DownloadRequest]) -> list[dict]:
    plan = []
    for req in requests:
        plan.append(
            {
                "request_id": make_request_id(req),
                "source_key": req.source_key,
                "variables": list(req.variables),
                "start_date": req.start_date,
                "end_date": req.end_date,
                "bbox": req.bbox,
                "target_dir": req.target_dir,
                "notes": req.notes,
                "frequency": req.frequency,
                "format_preference": req.format_preference,
            }
        )
    return plan
