from __future__ import annotations

from pathlib import Path

from premise.acquisition.base import DownloadRequest
from premise.acquisition.catalog import get_source
from premise.acquisition.registry import get_downloader_class_by_method


def download_dataset(request: DownloadRequest, **downloader_kwargs) -> list[Path]:
    source_config = get_source(request.source_key)
    downloader_cls = get_downloader_class_by_method(source_config.method)
    downloader = downloader_cls(source_config=source_config, **downloader_kwargs)
    return downloader.download(request)
