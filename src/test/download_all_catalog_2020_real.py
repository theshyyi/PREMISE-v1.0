from __future__ import annotations

"""
Real-download runner for every dataset defined in the acquisition catalog.

What this script does
---------------------
1. Loads the acquisition package from the local project.
2. Loads the catalog from either:
   - a sibling catalog.py placed next to this script, or
   - the project's own acquisition catalog module.
3. Iterates through every unique DataSource in DEFAULT_SOURCES.
4. Builds a 2020 download request for each source.
5. Tries to perform a real download into one root directory.
6. Writes a machine-readable report showing downloaded / skipped / failed.

Important behavior
------------------
- The script prefers the *catalog* as the source of truth.
- It supports mixed project states, for example:
  * catalog already contains ftp / earthdata_cmr entries,
  * but the installed project registry is still older.
- To handle that, the script uses:
  * project downloader classes when they are available, and
  * internal fallback executors for a few generic methods:
      - http_direct
      - ftp
      - earthdata_cmr
- Some methods are still intentionally skipped by default if they do not
  currently represent a true 2020-bounded real download in the installed
  workflow, for example ESGF search-only or manual portal-export flows.

This script is designed for direct execution in PyCharm or from the terminal.
All user parameters are configured in the block below.
"""

import csv
import ftplib
import importlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
import types
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests

# =============================================================================
# User-editable parameters
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Leave as None for automatic detection.
PROJECT_ROOT: Optional[Path] = None

# If a catalog.py is placed next to this script, the script will prefer it.
# Otherwise it falls back to the project's own acquisition catalog.
CATALOG_OVERRIDE_PATH: Optional[Path] = None

DOWNLOAD_ROOT_NAME = "acquisition_real_download_test_2020_full"
SUMMARY_BASENAME = "acquisition_real_download_2020_full"
YEAR = 2020
START_DATE = f"{YEAR}-01-01"
END_DATE = f"{YEAR}-12-31"

# For subsettable providers only. Do not force bbox onto all datasets.
USE_SMALL_BBOX_FOR_SUBSETTABLE = True
TEST_BBOX = (110.0, 30.0, 111.0, 31.0)
SUBSETTABLE_METHODS = {"cds_api", "erddap", "earthdata_cmr"}

# These methods are not guaranteed to enforce a true 2020 slice in the current
# workflow unless you have a source-specific implementation. Keep False unless
# you explicitly want them attempted.
ALLOW_NON_TEMPORAL_METHODS = False

# Portal export generally needs a pre-resolved export URL. ESGF search usually
# produces discovery or wget scripts rather than immediate data files.
ALLOW_PORTAL_EXPORT = False
ALLOW_ESGF_SEARCH_ONLY = False

TIMEOUT = 120
MAX_RETRIES = 2
SLEEP_SECONDS = 1.0
CHUNK_SIZE = 1024 * 1024
EARTHDATA_PAGE_SIZE = 500

# When a dataset has many variables, choose one representative variable.
PREFERRED_VARIABLES: Dict[str, str] = {
    "era5_single_levels_hourly": "total_precipitation",
    "era5land_hourly": "total_precipitation",
    "gleam_monthly_nc": "potential_evaporation",
    "cmip6_esgf": "pr",
    "cmip5_esgf": "pr",
    "merra2_tavg1_2d_flx_nx_hourly": "precipitation",
    "merra2_tavg1_2d_slv_nx_hourly": "temperature",
    "merra2_tavgm_2d_flx_nx_monthly": "precipitation",
    "gldas_noah025_3h_v21": "precipitation",
    "fldas_noah01_c_gl_m_v001": "precipitation",
    "fldas_noah01_cp_gl_m_v001": "precipitation",
}

# Optional format preferences for request.format_preference.
FORMAT_PREFERENCES: Dict[str, str] = {
    "era5_single_levels_hourly": "grib",
    "era5land_hourly": "grib",
    "persiann_portal": "nc",
    "persiann_ccs_portal": "nc",
    "persiann_cdr_portal": "nc",
    "persiann_ccs_cdr_v2_portal": "nc",
    "pdir_now_portal": "nc",
    "persiann_v3_portal": "nc",
    "persiann_cdr_v3_portal": "nc",
    "cmfd": "nc",
}

# Product-specific reality checks for YEAR=2020.
YEAR_UNAVAILABLE_RULES: Dict[str, str] = {
    "trmm_3b42_daily_v7": "TRMM TMPA ended in 2019; 2020 is not available for this product.",
    "trmm_3b42_3hourly_v7": "TRMM TMPA ended in 2019; 2020 is not available for this product.",
}

# =============================================================================
# Credential and external-tool inputs (environment-variable driven)
# =============================================================================

IMERG_USERNAME = os.getenv("IMERG_USERNAME", "lexinlong9@gmail.com")
IMERG_PASSWORD = os.getenv("IMERG_PASSWORD", "lexinlong9@gmail.com")

EARTHDATA_TOKEN = os.getenv("EARTHDATA_TOKEN", "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InhpbmxvbmdsZSIsImV4cCI6MTc4MTUxMjc2MiwiaWF0IjoxNzc2MzI4NzYyLCJpc3MiOiJodHRwczovL3Vycy5lYXJ0aGRhdGEubmFzYS5nb3YiLCJpZGVudGl0eV9wcm92aWRlciI6ImVkbF9vcHMiLCJhY3IiOiJlZGwiLCJhc3N1cmFuY2VfbGV2ZWwiOjN9.Q7Wp4_2qLS5hbqseW3mw0XZ81L4DhyendItcl0E1P9IQ11GBujHZHj9UYwvJl2-CabCi096JuNQGPkwbW2TMEypkNcZBCVqAlN4WdCXDi9-Km5Z1y8FqKocL2n-FPSVufAtf8eNzN6WbPI3sDrTMdhOYv1MfVjUrvpVc2tc_fHe8xkwcCfYJrqwxthRo1cX0ejefo_XH7meUOn0urxuLMFx8CJcdBFEIx8GKO8G0M0JeXPRVK0ZgyqhwB8A2VCcMIYLJWgxOtUfYQK1WC_x8qCGOsJRenoMXukNcKTVxvS7onfY55ir7LazU3f5wUCk2cFKa8n-_FuYHXun3T7B3hw") or os.getenv("EDL_TOKEN", "")
EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME", "xinlongle")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD", "@LeLebaron417914")

GLEAM_HOST = os.getenv("GLEAM_HOST", "hydras.ugent.be")
GLEAM_PORT = int(os.getenv("GLEAM_PORT", "2225"))
GLEAM_USERNAME = os.getenv("GLEAM_USERNAME", "gleamuser")
GLEAM_PASSWORD = os.getenv("GLEAM_PASSWORD", "GLEAM4#h-cel_924")
GLEAM_REMOTE_BASE_DIR = os.getenv("GLEAM_REMOTE_BASE_DIR", './data/v4.2b/monthly/Ep')

MSWEP_REMOTE_NAME = os.getenv("MSWEP_REMOTE_NAME", "")
MSWEP_REMOTE_PATH = os.getenv("MSWEP_REMOTE_PATH", "")
RCLONE_BINARY = os.getenv("RCLONE_BINARY", "rclone")

# Per-source manual portal export URL.
# Example env var name for source key "cmfd": ACQ_EXPORT_URL__CMFD
PORTAL_EXPORT_URLS: Dict[str, str] = {}

# ESGF facets are source-specific.
# Example structure:
ESGF_FACETS = {
    "cmip6_esgf": {
        "model": "MPI-ESM1-2-LR",
        "experiment": "historical",
        "member": "r1i1p1f1",
        "table": "day",
        "grid": "gn",
    }
}
ESGF_FACETS: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# Path / import bootstrap
# =============================================================================


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    candidates: List[Path] = []
    for p in [start] + list(start.parents):
        candidates.append(p)
        candidates.append(p / "src")

    seen: set[str] = set()
    for root in candidates:
        root = root.resolve()
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        if (root / "premise" / "acquisition").exists():
            return root
        if (root / "acquisition").exists():
            return root

    raise FileNotFoundError(
        "Could not auto-detect project root. Expected one of: "
        "<root>/premise/acquisition, <root>/acquisition, "
        "<root>/src/premise/acquisition, or <root>/src/acquisition."
    )


if PROJECT_ROOT is None:
    PROJECT_ROOT = find_project_root(SCRIPT_DIR)
PROJECT_ROOT = PROJECT_ROOT.resolve()
DOWNLOAD_ROOT = PROJECT_ROOT / DOWNLOAD_ROOT_NAME

if CATALOG_OVERRIDE_PATH is None:
    sibling_catalog = SCRIPT_DIR / "catalog.py"
    CATALOG_OVERRIDE_PATH = sibling_catalog if sibling_catalog.exists() else None


class BootstrapResult:
    def __init__(self, namespace: str, package_root: Path, DownloadRequest: Any) -> None:
        self.namespace = namespace
        self.package_root = package_root
        self.DownloadRequest = DownloadRequest


def bootstrap_acquisition_package(project_root: Path) -> BootstrapResult:
    root = project_root.resolve()
    premise_pkg = root / "premise" / "acquisition"
    acq_pkg = root / "acquisition"

    if premise_pkg.exists():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        mod = importlib.import_module("premise.acquisition.base")
        return BootstrapResult("premise.acquisition", premise_pkg, getattr(mod, "DownloadRequest"))

    if acq_pkg.exists():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        if "premise" not in sys.modules:
            premise = types.ModuleType("premise")
            premise.__path__ = [str(root)]
            sys.modules["premise"] = premise
        if "premise.acquisition" not in sys.modules:
            acquisition_pkg = types.ModuleType("premise.acquisition")
            acquisition_pkg.__path__ = [str(acq_pkg)]
            sys.modules["premise.acquisition"] = acquisition_pkg
            setattr(sys.modules["premise"], "acquisition", acquisition_pkg)

        mod = importlib.import_module("premise.acquisition.base")
        return BootstrapResult("premise.acquisition", acq_pkg, getattr(mod, "DownloadRequest"))

    raise FileNotFoundError(
        f"Could not find acquisition package under project root: {root}"
    )


BOOT = bootstrap_acquisition_package(PROJECT_ROOT)


def unique_sources(default_sources: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for src in default_sources:
        key = getattr(src, "key").lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(src)
    return out

def load_catalog_sources(override_path: Optional[Path]) -> tuple[list[Any], str]:
    if override_path is not None and Path(override_path).exists():
        spec = importlib.util.spec_from_file_location("_catalog_override_module", str(override_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load catalog override: {override_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        default_sources = getattr(module, "DEFAULT_SOURCES")
        return unique_sources(default_sources), f"override:{Path(override_path).resolve()}"

    catalog_mod = importlib.import_module("premise.acquisition.catalog")
    default_sources = getattr(catalog_mod, "DEFAULT_SOURCES")
    return unique_sources(default_sources), "project_module"


SOURCES, CATALOG_SOURCE = load_catalog_sources(CATALOG_OVERRIDE_PATH)

# =============================================================================
# Utilities
# =============================================================================





def choose_variable(src: Any) -> str:
    return PREFERRED_VARIABLES.get(src.key, src.variables[0])


def choose_bbox(src: Any) -> Optional[Tuple[float, float, float, float]]:
    if not USE_SMALL_BBOX_FOR_SUBSETTABLE:
        return None
    if src.method.lower().strip() in SUBSETTABLE_METHODS:
        return TEST_BBOX
    return None


KNOWN_METHOD_CLASS_IMPORTS: Dict[str, tuple[str, str]] = {
    "http_direct": ("premise.acquisition.downloaders.http_direct", "HTTPDirectDownloader"),
    "http_listing": ("premise.acquisition.downloaders.http_listing", "HTTPListingDownloader"),
    "erddap": ("premise.acquisition.downloaders.erddap", "ERDDAPDownloader"),
    "cds_api": ("premise.acquisition.downloaders.cds_api", "CDSAPIDownloader"),
    "sftp": ("premise.acquisition.downloaders.sftp", "SFTPDownloader"),
    "ftp": ("premise.acquisition.downloaders.ftp", "FTPDownloader"),
    "earthdata_cmr": ("premise.acquisition.downloaders.earthdata_cmr", "EarthdataCMRDownloader"),
    "rclone": ("premise.acquisition.downloaders.rclone", "RcloneDownloader"),
    "portal_export": ("premise.acquisition.downloaders.portal_export", "PortalExportDownloader"),
    "esgf": ("premise.acquisition.downloaders.esgf", "ESGFSearchDownloader"),
    "esgf_wget": ("premise.acquisition.downloaders.wget", "ESGFWgetDownloader"),
}


def resolve_method_registry() -> dict[str, Any]:
    registry: dict[str, Any] = {}
    try:
        reg_mod = importlib.import_module("premise.acquisition.registry")
        method_registry = getattr(reg_mod, "METHOD_REGISTRY", {})
        for key, cls in method_registry.items():
            registry[str(key).lower().strip()] = cls
    except Exception:
        pass

    for method, (modname, classname) in KNOWN_METHOD_CLASS_IMPORTS.items():
        if method in registry:
            continue
        try:
            mod = importlib.import_module(modname)
            registry[method] = getattr(mod, classname)
        except Exception:
            continue
    return registry


METHOD_REGISTRY = resolve_method_registry()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_files(paths: Iterable[Path]) -> Dict[str, Any]:
    existing = [Path(p) for p in paths if Path(p).exists()]
    total_size = sum(p.stat().st_size for p in existing)
    return {
        "file_count": len(existing),
        "total_size_bytes": total_size,
        "files": [str(p) for p in existing],
    }


def safe_src_dict(src: Any) -> Dict[str, Any]:
    if is_dataclass(src):
        return asdict(src)
    out = {}
    for name in (
        "key", "title", "variables", "temporal_resolution", "spatial_resolution",
        "provider", "method", "params", "aliases", "format_hints", "notes",
        "official_url", "access_mode", "auth_required", "product_notes",
    ):
        if hasattr(src, name):
            out[name] = getattr(src, name)
    return out

# =============================================================================
# Fallback helpers for newer generic methods
# =============================================================================


def _expand_years(start_date: str, end_date: str) -> List[int]:
    ys = int(start_date.split("-")[0])
    ye = int(end_date.split("-")[0])
    return list(range(ys, ye + 1))


def _expand_dates(start_date: str, end_date: str) -> List[datetime]:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError("end_date must not be earlier than start_date")
    out: List[datetime] = []
    dt = start_dt
    while dt <= end_dt:
        out.append(dt)
        dt += timedelta(days=1)
    return out


def _expand_decade_blocks(start_date: str, end_date: str, anchor_year: int, block_length: int) -> List[tuple[int, int]]:
    ys = int(start_date.split("-")[0])
    ye = int(end_date.split("-")[0])
    blocks: List[tuple[int, int]] = []

    # Move anchor backward if needed.
    current_start = anchor_year
    while current_start > ys:
        current_start -= block_length

    while current_start <= ye:
        current_end = current_start + block_length - 1
        if current_end >= ys and current_start <= ye:
            blocks.append((current_start, current_end))
        current_start += block_length
    return blocks


def fallback_http_direct_download(src: Any, request: Any) -> List[Path]:
    if request.target_dir is None:
        raise ValueError("target_dir must be provided.")
    if request.start_date is None or request.end_date is None:
        raise ValueError("start_date and end_date must be provided.")

    params = src.params
    base_url = params["base_url"]
    filename_pattern = params["filename_pattern"]
    directory_pattern = params.get("directory_pattern", "")
    time_expansion = params.get("time_expansion", "none")
    outdir = Path(request.target_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    items: List[Any]
    if time_expansion == "year":
        items = _expand_years(request.start_date, request.end_date)
    elif time_expansion == "date":
        items = _expand_dates(request.start_date, request.end_date)
    elif time_expansion == "none":
        items = [None]
    elif time_expansion == "decade":
        anchor = int(params.get("decade_anchor_year", 1901))
        length = int(params.get("decade_length", 10))
        items = _expand_decade_blocks(request.start_date, request.end_date, anchor, length)
    else:
        raise ValueError(f"Unsupported http_direct time_expansion: {time_expansion}")

    session = requests.Session()
    downloaded: List[Path] = []

    for item in items:
        if time_expansion == "year":
            filename = filename_pattern.format(year=item)
            subdir = directory_pattern.format(year=item) if directory_pattern else ""
        elif time_expansion == "date":
            dt: datetime = item
            year = f"{dt.year:04d}"
            month = f"{dt.month:02d}"
            day = f"{dt.day:02d}"
            date = f"{year}{month}{day}"
            filename = filename_pattern.format(year=year, month=month, day=day, date=date)
            subdir = directory_pattern.format(year=year, month=month, day=day, date=date) if directory_pattern else ""
        elif time_expansion == "decade":
            start_year, end_year = item
            filename = filename_pattern.format(start_year=start_year, end_year=end_year)
            subdir = directory_pattern.format(start_year=start_year, end_year=end_year) if directory_pattern else ""
        else:
            filename = filename_pattern
            subdir = directory_pattern if directory_pattern else ""

        url = urljoin(base_url, f"{subdir}{filename}")
        local_path = outdir / Path(filename).name
        if local_path.exists() and local_path.stat().st_size > 0:
            downloaded.append(local_path)
            continue

        success = False
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with session.get(url, stream=True, timeout=TIMEOUT) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                downloaded.append(local_path)
                success = True
                break
            except Exception as e:
                last_error = str(e)
                if local_path.exists():
                    try:
                        local_path.unlink()
                    except Exception:
                        pass
                if attempt < MAX_RETRIES:
                    time.sleep(SLEEP_SECONDS)
        if not success:
            raise RuntimeError(f"HTTPDirect fallback failed for {url}: {last_error}")

    return downloaded


def _ftp_expand_contexts(start_date: str, end_date: str, time_expansion: str) -> List[Dict[str, str]]:
    if time_expansion == "year":
        return [{"year": f"{y:04d}"} for y in _expand_years(start_date, end_date)]
    if time_expansion == "date":
        out: List[Dict[str, str]] = []
        for dt in _expand_dates(start_date, end_date):
            year = f"{dt.year:04d}"
            month = f"{dt.month:02d}"
            day = f"{dt.day:02d}"
            out.append({
                "year": year,
                "month": month,
                "day": day,
                "date": f"{year}{month}{day}",
            })
        return out
    if time_expansion == "none":
        return [{}]
    raise ValueError(f"Unsupported ftp time_expansion: {time_expansion}")


def _ftp_join(base_dir: str, name: str) -> str:
    left = (base_dir or "").rstrip("/")
    right = (name or "").lstrip("/")
    if not left:
        return "/" + right if right else "/"
    return f"{left}/{right}" if right else left


def fallback_ftp_download(src: Any, request: Any) -> List[Path]:
    if request.target_dir is None:
        raise ValueError("target_dir must be provided.")
    if request.start_date is None or request.end_date is None:
        raise ValueError("start_date and end_date must be provided.")

    params = src.params
    host = params.get("host")
    if not host:
        raise ValueError("FTP source requires params['host'].")
    port = int(params.get("default_port", 21))
    username = params.get("default_username", "anonymous")
    password = params.get("default_password", "anonymous@")
    use_tls = bool(params.get("use_tls", False))
    base_dir = (params.get("remote_base_dir", "") or "").rstrip("/")
    time_expansion = params.get("time_expansion", "date")
    directory_pattern = params.get("directory_pattern", "")
    filename_pattern = params.get("filename_pattern")
    file_regex = re.compile(params["file_regex"], re.IGNORECASE) if params.get("file_regex") else None
    contains_cfg = params.get("contains", []) or []
    if isinstance(contains_cfg, str):
        contains_cfg = [contains_cfg]
    contains_template_cfg = params.get("contains_template", []) or []
    if isinstance(contains_template_cfg, str):
        contains_template_cfg = [contains_template_cfg]

    outdir = Path(request.target_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ftp = ftplib.FTP_TLS(timeout=TIMEOUT) if use_tls else ftplib.FTP(timeout=TIMEOUT)
    ftp.connect(host=host, port=port, timeout=TIMEOUT)
    ftp.login(user=username, passwd=password)
    if use_tls:
        ftp.prot_p()
    ftp.set_pasv(True)

    discovered: List[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        for ctx in _ftp_expand_contexts(request.start_date, request.end_date, time_expansion):
            remote_dir = _ftp_join(base_dir, directory_pattern.format(**ctx) if directory_pattern else "")
            if filename_pattern:
                filename = filename_pattern.format(**ctx)
                remote_path = _ftp_join(remote_dir, filename)
                if remote_path not in seen:
                    seen.add(remote_path)
                    discovered.append((remote_path, Path(filename).name))
                continue

            names = ftp.nlst(remote_dir)
            formatted_contains = [s.format(**ctx).lower() for s in contains_template_cfg]
            static_contains = [str(s).lower() for s in contains_cfg]
            required_tokens = static_contains + formatted_contains

            for remote_path in names:
                name = Path(remote_path).name
                if file_regex and not file_regex.search(name):
                    continue
                lower_name = name.lower()
                if required_tokens and not all(tok in lower_name for tok in required_tokens):
                    continue
                if remote_path not in seen:
                    seen.add(remote_path)
                    discovered.append((remote_path, name))

        downloaded: List[Path] = []
        for remote_path, name in discovered:
            local_path = outdir / name
            if local_path.exists() and local_path.stat().st_size > 0:
                downloaded.append(local_path)
                continue
            success = False
            last_error = ""
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    with open(local_path, "wb") as f:
                        ftp.retrbinary(f"RETR {remote_path}", f.write, blocksize=CHUNK_SIZE)
                    downloaded.append(local_path)
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    if local_path.exists():
                        try:
                            local_path.unlink()
                        except Exception:
                            pass
                    if attempt < MAX_RETRIES:
                        time.sleep(SLEEP_SECONDS)
            if not success:
                raise RuntimeError(f"FTP fallback failed for ftp://{host}:{port}{remote_path}: {last_error}")

        return downloaded
    finally:
        try:
            ftp.quit()
        except Exception:
            try:
                ftp.close()
            except Exception:
                pass


def _earthdata_auth_session() -> requests.Session:
    session = requests.Session()
    session.headers.setdefault("User-Agent", "premise-acquisition-real-test")
    if EARTHDATA_TOKEN:
        session.headers["Authorization"] = f"Bearer {EARTHDATA_TOKEN}"
    elif EARTHDATA_USERNAME and EARTHDATA_PASSWORD:
        session.auth = (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
    else:
        raise RuntimeError(
            "Earthdata credentials are missing. Set EARTHDATA_TOKEN or "
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD."
        )
    return session


def _earthdata_build_query(src: Any, request: Any, page_num: int, page_size: int) -> dict[str, Any]:
    params = src.params
    query: dict[str, Any] = {
        "page_size": page_size,
        "page_num": page_num,
        "sort_key[]": "+start_date",
        "temporal": f"{request.start_date}T00:00:00Z,{request.end_date}T23:59:59Z",
    }
    if params.get("concept_id"):
        query["concept_id"] = params["concept_id"]
    if params.get("short_name"):
        query["short_name"] = params["short_name"]
    if params.get("version"):
        query["version"] = str(params["version"])
    if params.get("provider"):
        query["provider"] = params["provider"]
    if request.bbox is not None:
        west, south, east, north = request.bbox
        query["bounding_box"] = f"{west},{south},{east},{north}"
    return query


def _earthdata_is_probable_data_link(link: dict, href: str) -> bool:
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


def _earthdata_extract_links(entry: dict, params: dict) -> list[tuple[str, str]]:
    links = entry.get("links", []) or []
    preferred = [s.lower() for s in params.get("preferred_link_substrings", [])]
    excluded = [s.lower() for s in params.get("exclude_link_substrings", [])]
    allow_s3 = bool(params.get("allow_s3_links", False))
    filename_regex = re.compile(params["filename_regex"], re.IGNORECASE) if params.get("filename_regex") else None

    hits: list[tuple[str, str]] = []
    for link in links:
        href = (link or {}).get("href")
        if not href or link.get("inherited"):
            continue
        href_lower = href.lower()
        if not allow_s3 and href_lower.startswith("s3://"):
            continue
        if any(tok in href_lower for tok in excluded):
            continue
        if not _earthdata_is_probable_data_link(link, href):
            continue
        if preferred and not any(tok in href_lower for tok in preferred):
            continue
        filename = Path(urlparse(href).path).name or "earthdata_download.bin"
        if filename_regex and not filename_regex.search(filename):
            continue
        hits.append((href, filename))
    return hits


def fallback_earthdata_cmr_download(src: Any, request: Any) -> List[Path]:
    if request.target_dir is None:
        raise ValueError("target_dir must be provided.")
    if request.start_date is None or request.end_date is None:
        raise ValueError("start_date and end_date must be provided.")

    session = _earthdata_auth_session()
    params = src.params
    cmr_url = params.get("cmr_granules_url", "https://cmr.earthdata.nasa.gov/search/granules.json")
    outdir = Path(request.target_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    discovered: List[tuple[str, str]] = []
    seen_urls: set[str] = set()
    page_num = 1
    while True:
        query = _earthdata_build_query(src, request, page_num=page_num, page_size=EARTHDATA_PAGE_SIZE)
        r = session.get(cmr_url, params=query, timeout=TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        entries = payload.get("feed", {}).get("entry", [])
        if not entries:
            break
        for entry in entries:
            for url, filename in _earthdata_extract_links(entry, params):
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                discovered.append((url, filename))
        if len(entries) < EARTHDATA_PAGE_SIZE:
            break
        page_num += 1

    downloaded: List[Path] = []
    for url, filename in discovered:
        local_path = outdir / filename
        if local_path.exists() and local_path.stat().st_size > 0:
            downloaded.append(local_path)
            continue
        success = False
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with session.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                downloaded.append(local_path)
                success = True
                break
            except Exception as e:
                last_error = str(e)
                if local_path.exists():
                    try:
                        local_path.unlink()
                    except Exception:
                        pass
                if attempt < MAX_RETRIES:
                    time.sleep(SLEEP_SECONDS)
        if not success:
            raise RuntimeError(f"Earthdata CMR fallback failed for {url}: {last_error}")
    return downloaded

# =============================================================================
# Request, policy, and execution helpers
# =============================================================================


def build_request(src: Any, outdir: Path) -> Any:
    return BOOT.DownloadRequest(
        source_key=src.key,
        variables=(choose_variable(src),),
        start_date=START_DATE,
        end_date=END_DATE,
        bbox=choose_bbox(src),
        target_dir=str(outdir),
        notes="real acquisition test for catalog-defined sources in 2020",
        frequency=getattr(src, "temporal_resolution", None),
        format_preference=FORMAT_PREFERENCES.get(src.key),
    )


def portal_export_env_name(source_key: str) -> str:
    return f"ACQ_EXPORT_URL__{source_key.upper()}"


def skip_reason_for_source(src: Any) -> Optional[str]:
    key = src.key.lower().strip()
    method = src.method.lower().strip()

    if key in YEAR_UNAVAILABLE_RULES:
        return YEAR_UNAVAILABLE_RULES[key]

    if method in {"sftp", "rclone"} and not ALLOW_NON_TEMPORAL_METHODS:
        return "non-temporal/manual downloader disabled by ALLOW_NON_TEMPORAL_METHODS=False"

    if method == "portal_export" and not ALLOW_PORTAL_EXPORT:
        return "portal_export disabled by ALLOW_PORTAL_EXPORT=False"

    if method == "esgf" and not ALLOW_ESGF_SEARCH_ONLY:
        return "current ESGF downloader is discovery/search-oriented, not a direct real-data download"

    if method == "http_listing" and key == "imerg_final_hdf5":
        if not (IMERG_USERNAME and IMERG_PASSWORD):
            return "missing IMERG_USERNAME or IMERG_PASSWORD"

    if method == "sftp":
        if not (GLEAM_HOST and GLEAM_USERNAME and GLEAM_PASSWORD):
            return "missing GLEAM_HOST / GLEAM_USERNAME / GLEAM_PASSWORD"

    if method == "rclone":
        if not (MSWEP_REMOTE_NAME and MSWEP_REMOTE_PATH):
            return "missing MSWEP_REMOTE_NAME or MSWEP_REMOTE_PATH"

    if method == "portal_export":
        export_url = os.getenv(portal_export_env_name(src.key), "")
        if not export_url:
            return f"missing {portal_export_env_name(src.key)}"

    if method == "earthdata_cmr":
        if not (EARTHDATA_TOKEN or (EARTHDATA_USERNAME and EARTHDATA_PASSWORD)):
            return "missing EARTHDATA_TOKEN or EARTHDATA_USERNAME/EARTHDATA_PASSWORD"

    return None


def build_module_downloader_kwargs(src: Any) -> Dict[str, Any]:
    method = src.method.lower().strip()
    kwargs: Dict[str, Any] = {
        "timeout": TIMEOUT,
        "max_retries": MAX_RETRIES,
        "sleep_seconds": SLEEP_SECONDS,
    }

    if method in {"http_direct", "http_listing", "erddap", "portal_export", "ftp", "earthdata_cmr"}:
        kwargs["chunk_size"] = CHUNK_SIZE

    if method == "http_listing" and src.key == "imerg_final_hdf5":
        kwargs["username"] = IMERG_USERNAME
        kwargs["password"] = IMERG_PASSWORD

    if method == "sftp":
        kwargs.update({
            "host": GLEAM_HOST,
            "port": GLEAM_PORT,
            "username": GLEAM_USERNAME,
            "password": GLEAM_PASSWORD,
        })
        if GLEAM_REMOTE_BASE_DIR:
            kwargs["remote_base_dir"] = GLEAM_REMOTE_BASE_DIR

    if method == "rclone":
        kwargs.update({
            "remote_name": MSWEP_REMOTE_NAME,
            "remote_path": MSWEP_REMOTE_PATH,
            "rclone_binary": RCLONE_BINARY,
        })

    if method == "portal_export":
        kwargs["export_url"] = os.getenv(portal_export_env_name(src.key), "")

    if method == "esgf":
        facets = ESGF_FACETS.get(src.key)
        if facets:
            kwargs["facets"] = facets

    if method == "earthdata_cmr":
        if EARTHDATA_TOKEN:
            kwargs["earthdata_token"] = EARTHDATA_TOKEN
        elif EARTHDATA_USERNAME and EARTHDATA_PASSWORD:
            kwargs["username"] = EARTHDATA_USERNAME
            kwargs["password"] = EARTHDATA_PASSWORD

    return kwargs


def choose_executor(src: Any) -> tuple[str, Any]:
    """
    Returns
    -------
    backend_name : str
        One of: module, fallback_http_direct, fallback_ftp, fallback_earthdata_cmr
    executor : callable or downloader class
    """
    method = src.method.lower().strip()
    params = getattr(src, "params", {}) or {}

    if method == "http_direct":
        # Prefer fallback for modern catalog cases like none / decade even when the
        # installed module may still only support year/date.
        if params.get("time_expansion") in {"none", "decade"}:
            return "fallback_http_direct", fallback_http_direct_download
        # If module class exists, use it; otherwise fallback.
        cls = METHOD_REGISTRY.get("http_direct")
        return ("module", cls) if cls is not None else ("fallback_http_direct", fallback_http_direct_download)

    if method == "ftp":
        cls = METHOD_REGISTRY.get("ftp")
        return ("module", cls) if cls is not None else ("fallback_ftp", fallback_ftp_download)

    if method == "earthdata_cmr":
        cls = METHOD_REGISTRY.get("earthdata_cmr")
        return ("module", cls) if cls is not None else ("fallback_earthdata_cmr", fallback_earthdata_cmr_download)

    cls = METHOD_REGISTRY.get(method)
    if cls is not None:
        return "module", cls

    return "missing", None


def execute_source_download(src: Any, request: Any) -> tuple[str, List[Path]]:
    backend_name, executor = choose_executor(src)
    if backend_name == "missing" or executor is None:
        raise KeyError(f"No downloader available for method='{src.method}' in the current project.")

    if backend_name == "module":
        downloader = executor(source_config=src, **build_module_downloader_kwargs(src))
        return backend_name, downloader.download(request)

    return backend_name, executor(src, request)

# =============================================================================
# Main execution
# =============================================================================


def main() -> None:
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Real acquisition download test for all catalog-defined datasets")
    print(f"SCRIPT_DIR       : {SCRIPT_DIR}")
    print(f"PROJECT_ROOT     : {PROJECT_ROOT}")
    print(f"CATALOG_SOURCE   : {CATALOG_SOURCE}")
    print(f"DOWNLOAD_ROOT    : {DOWNLOAD_ROOT}")
    print(f"DATE_RANGE       : {START_DATE} -> {END_DATE}")
    print(f"BBOX_MODE        : {'enabled for subsettable methods' if USE_SMALL_BBOX_FOR_SUBSETTABLE else 'disabled'}")
    print(f"ALLOW_NON_TEMPORAL_METHODS : {ALLOW_NON_TEMPORAL_METHODS}")
    print(f"ALLOW_PORTAL_EXPORT        : {ALLOW_PORTAL_EXPORT}")
    print(f"ALLOW_ESGF_SEARCH_ONLY     : {ALLOW_ESGF_SEARCH_ONLY}")
    print("=" * 96)

    dataset_rows: List[Dict[str, Any]] = []

    for src in SOURCES:
        src_dir = DOWNLOAD_ROOT / src.key
        src_dir.mkdir(parents=True, exist_ok=True)
        request = build_request(src, src_dir)

        row: Dict[str, Any] = {
            "source_key": src.key,
            "title": src.title,
            "provider": src.provider,
            "method": src.method,
            "auth_required": src.auth_required,
            "variable": choose_variable(src),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "bbox": request.bbox,
            "target_dir": str(src_dir),
            "status": "pending",
            "message": "",
            "executor": "",
            "file_count": 0,
            "total_size_bytes": 0,
            "catalog_source": CATALOG_SOURCE,
        }

        skip_reason = skip_reason_for_source(src)
        if skip_reason is not None:
            row["status"] = "skipped"
            row["message"] = skip_reason
            row["executor"] = "not_run"
            dataset_rows.append(row)
            print(f"[SKIP] {src.key}: {skip_reason}")
            continue

        try:
            executor_name, outputs = execute_source_download(src, request)
            stats = summarize_files(outputs)
            row["executor"] = executor_name
            row["file_count"] = stats["file_count"]
            row["total_size_bytes"] = stats["total_size_bytes"]
            row["files_json"] = json.dumps(stats["files"], ensure_ascii=False)
            if stats["file_count"] > 0:
                row["status"] = "downloaded"
                row["message"] = "ok"
                print(f"[ OK ] {src.key} [{executor_name}] -> {stats['file_count']} file(s), {stats['total_size_bytes']} bytes")
            else:
                row["status"] = "no_files_returned"
                row["message"] = "downloader returned no files"
                print(f"[WARN] {src.key} [{executor_name}] -> no files returned")
        except Exception as e:
            row["status"] = "failed"
            row["message"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()
            print(f"[FAIL] {src.key}: {type(e).__name__}: {e}")

        dataset_rows.append(row)

    summary = {
        "script_dir": str(SCRIPT_DIR),
        "project_root": str(PROJECT_ROOT),
        "catalog_source": CATALOG_SOURCE,
        "download_root": str(DOWNLOAD_ROOT),
        "year": YEAR,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "bbox": TEST_BBOX if USE_SMALL_BBOX_FOR_SUBSETTABLE else None,
        "dataset_count": len(dataset_rows),
        "downloaded": sum(1 for r in dataset_rows if r["status"] == "downloaded"),
        "skipped": sum(1 for r in dataset_rows if r["status"] == "skipped"),
        "failed": sum(1 for r in dataset_rows if r["status"] == "failed"),
        "no_files_returned": sum(1 for r in dataset_rows if r["status"] == "no_files_returned"),
    }

    summary_json = DOWNLOAD_ROOT / f"{SUMMARY_BASENAME}_summary.json"
    datasets_json = DOWNLOAD_ROOT / f"{SUMMARY_BASENAME}_datasets.json"
    datasets_csv = DOWNLOAD_ROOT / f"{SUMMARY_BASENAME}_datasets.csv"
    sources_json = DOWNLOAD_ROOT / f"{SUMMARY_BASENAME}_catalog_snapshot.json"

    write_json(summary_json, summary)
    write_json(datasets_json, dataset_rows)
    write_json(sources_json, [safe_src_dict(src) for src in SOURCES])
    write_csv(datasets_csv, dataset_rows)

    print("-" * 96)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {summary_json}")
    print(f"Saved: {datasets_json}")
    print(f"Saved: {datasets_csv}")
    print(f"Saved: {sources_json}")


if __name__ == "__main__":
    main()
