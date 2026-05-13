from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from premise.acquisition.base import BaseDownloader, DownloadRequest, make_request_id
from premise.acquisition.catalog import DataSource
from premise.acquisition.logging_utils import AcquisitionRunRecorder, now_str


class EarthdataCMRDownloader(BaseDownloader):
    """
    Generic downloader for NASA Earthdata collections discovered via the CMR
    granules API and downloaded over HTTPS.

    Source params
    -------------
    One of the following identifiers should be provided:
    - concept_id
    - short_name (optionally with version)

    Optional source params
    ----------------------
    cmr_granules_url : str
    provider : str
    file_limit_default : int | None
    preferred_link_substrings : list[str]
    exclude_link_substrings : list[str]
    filename_regex : str
    allow_s3_links : bool
    """

    provider_name = "EarthdataCMR"

    def __init__(
        self,
        source_config: DataSource,
        timeout: int = 120,
        chunk_size: int = 1024 * 1024,
        max_retries: int = 3,
        sleep_seconds: float = 1.0,
        page_size: int = 2000,
        session: Optional[requests.Session] = None,
        earthdata_token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.source_config = source_config
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.page_size = page_size
        self.session = session or requests.Session()
        self.earthdata_token = earthdata_token or os.getenv("EARTHDATA_TOKEN") or os.getenv("EDL_TOKEN")
        self.username = username or os.getenv("EARTHDATA_USERNAME")
        self.password = password or os.getenv("EARTHDATA_PASSWORD")

        self.session.headers.setdefault("User-Agent", "premise-acquisition/earthdata-cmr")
        if self.earthdata_token:
            self.session.headers["Authorization"] = f"Bearer {self.earthdata_token}"
        elif self.username and self.password:
            self.session.auth = (self.username, self.password)

    def download(self, request: DownloadRequest) -> list[Path]:
        src = self.source_config
        if request.target_dir is None:
            raise ValueError("target_dir must be provided.")
        if request.start_date is None or request.end_date is None:
            raise ValueError("start_date and end_date must be provided.")

        outdir = Path(request.target_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        request_id = make_request_id(request)
        params = src.params
        cmr_url = params.get("cmr_granules_url", "https://cmr.earthdata.nasa.gov/search/granules.json")
        file_limit = params.get("file_limit_default")

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
                "cmr_granules_url": cmr_url,
                "concept_id": params.get("concept_id"),
                "short_name": params.get("short_name"),
                "version": params.get("version"),
            },
        )

        discovered = self._search_granule_links(request=request, recorder=recorder)
        if file_limit is not None:
            discovered = discovered[: int(file_limit)]

        downloaded: list[Path] = []
        for url, filename in discovered:
            local_path = outdir / filename
            if local_path.exists() and local_path.stat().st_size > 0:
                downloaded.append(local_path)
                recorder.record_skipped_existing(url=url, local_path=str(local_path), size_bytes=local_path.stat().st_size)
                continue

            success = False
            last_error = ""
            last_error_type = ""
            last_http_status = None
            for attempt in range(1, self.max_retries + 1):
                t0 = time.perf_counter()
                started = now_str()
                recorder.log_event("INFO", f"url={url} attempt={attempt} download_started")
                try:
                    with self.session.get(url, stream=True, timeout=self.timeout, allow_redirects=True) as r:
                        last_http_status = r.status_code
                        r.raise_for_status()
                        with open(local_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=self.chunk_size):
                                if chunk:
                                    f.write(chunk)
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
                        http_status=last_http_status,
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
                        http_status=last_http_status,
                        error_type=last_error_type,
                        error_message=last_error,
                        file_exists_after_download=local_path.exists(),
                    )
                    if attempt < self.max_retries:
                        time.sleep(self.sleep_seconds)
            if not success:
                print(f"[FAILED] {url} -> {last_error}")

        recorder.finalize(outdir)
        return downloaded

    def _search_granule_links(self, *, request: DownloadRequest, recorder: AcquisitionRunRecorder) -> list[tuple[str, str]]:
        params = self.source_config.params
        cmr_url = params.get("cmr_granules_url", "https://cmr.earthdata.nasa.gov/search/granules.json")
        page_size = int(params.get("cmr_page_size", self.page_size))

        page_num = 1
        discovered: list[tuple[str, str]] = []
        seen_urls: set[str] = set()

        while True:
            query = self._build_cmr_query_params(request=request, page_num=page_num, page_size=page_size)
            recorder.log_event("INFO", f"cmr_query_started page_num={page_num} params={query}")
            response = self.session.get(cmr_url, params=query, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            entries = payload.get("feed", {}).get("entry", [])
            if not entries:
                break

            page_hits = 0
            for entry in entries:
                for url, filename in self._extract_download_links_from_entry(entry=entry, params=params):
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    discovered.append((url, filename))
                    page_hits += 1

            if len(entries) < page_size:
                break
            page_num += 1

        return discovered

    def _build_cmr_query_params(self, *, request: DownloadRequest, page_num: int, page_size: int) -> dict[str, str | int]:
        src_params = self.source_config.params
        query: dict[str, str | int] = {
            "page_size": page_size,
            "page_num": page_num,
            "sort_key[]": "+start_date",
        }
        if src_params.get("concept_id"):
            query["concept_id"] = src_params["concept_id"]
        if src_params.get("short_name"):
            query["short_name"] = src_params["short_name"]
        if src_params.get("version"):
            query["version"] = str(src_params["version"])
        if src_params.get("provider"):
            query["provider"] = src_params["provider"]

        start_token = f"{request.start_date}T00:00:00Z"
        end_token = f"{request.end_date}T23:59:59Z"
        query["temporal"] = f"{start_token},{end_token}"

        if request.bbox is not None:
            west, south, east, north = request.bbox
            query["bounding_box"] = f"{west},{south},{east},{north}"
        return query

    @classmethod
    def _extract_download_links_from_entry(cls, *, entry: dict, params: dict) -> list[tuple[str, str]]:
        links = entry.get("links", []) or []
        preferred_substrings = [s.lower() for s in params.get("preferred_link_substrings", [])]
        exclude_substrings = [s.lower() for s in params.get("exclude_link_substrings", [])]
        allow_s3 = bool(params.get("allow_s3_links", False))
        filename_regex = re.compile(params["filename_regex"], re.IGNORECASE) if params.get("filename_regex") else None

        hits: list[tuple[str, str]] = []
        for link in links:
            href = (link or {}).get("href")
            if not href:
                continue
            if link.get("inherited"):
                continue
            href_lower = href.lower()
            if not allow_s3 and href_lower.startswith("s3://"):
                continue
            if any(token in href_lower for token in exclude_substrings):
                continue
            if not cls._is_probable_data_link(link, href=href):
                continue
            if preferred_substrings and not any(token in href_lower for token in preferred_substrings):
                continue

            filename = cls._filename_from_href(href)
            if filename_regex and not filename_regex.search(filename):
                continue
            hits.append((href, filename))
        return hits

    @staticmethod
    def _filename_from_href(href: str) -> str:
        parsed = urlparse(href)
        name = Path(parsed.path).name
        if name:
            return name
        return href.rstrip("/").split("/")[-1] or "earthdata_download.bin"

    @staticmethod
    def _is_probable_data_link(link: dict, *, href: str) -> bool:
        href_lower = href.lower()
        rel = str(link.get("rel", "")).lower()
        title = str(link.get("title", "")).lower()
        typ = str(link.get("type", "")).lower()

        if any(href_lower.endswith(ext) for ext in (".xml", ".iso", ".html", ".json", ".jpg", ".png", ".gif", ".md5", ".sha1", ".cmr.xml")):
            return False
        if any(token in href_lower for token in ("opendap", "docs", "metadata", "browse")):
            return False
        if any(token in rel for token in ("browse#", "metadata#", "service#")):
            return False
        if any(token in title for token in ("opendap", "browse", "metadata", "documentation")):
            return False
        if any(token in typ for token in ("text/html", "application/xml", "application/json")):
            return False
        return href.startswith(("https://", "http://", "s3://"))
