from __future__ import annotations

"""
Real-download smoke test for all datasets defined in the acquisition module.

This version distinguishes between:
1. spatially subsettable sources: pass bbox into DownloadRequest
2. non-subsettable/global-file sources: do NOT pass bbox, only constrain by time

Rationale
---------
Many products in the current acquisition module are distributed as whole files
(global or large regional tiles), so forcing bbox into every request is not
appropriate. In the current codebase, only a subset of methods really consume
`request.bbox` (primarily `cds_api` and `erddap`). Others either ignore it or
cannot support it at provider side.
"""

import csv
import json
import os
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Path discovery
# =============================================================================

def find_project_root(start: Path) -> Path:
    """
    Find the real import root automatically.

    It supports common layouts such as:
      - <root>/premise/acquisition
      - <root>/acquisition
      - <root>/src/premise/acquisition
      - <root>/src/acquisition

    This avoids hard-coding PROJECT_ROOT when the test script is placed under
    src/test or another subdirectory.
    """
    start = start.resolve()
    candidates: List[Path] = []
    for p in [start] + list(start.parents):
        candidates.append(p)
        candidates.append(p / "src")

    seen = set()
    for root in candidates:
        root = root.resolve()
        if str(root) in seen:
            continue
        seen.add(str(root))
        if (root / "premise" / "acquisition").exists():
            return root
        if (root / "acquisition").exists():
            return root

    raise FileNotFoundError(
        "Could not auto-detect project root from script location. "
        "Expected one of: <root>/premise/acquisition, <root>/acquisition, "
        "<root>/src/premise/acquisition, or <root>/src/acquisition."
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)

# =============================================================================
# User-editable parameters
# =============================================================================
DOWNLOAD_ROOT = PROJECT_ROOT / "acquisition_real_download_test_2020"
SUMMARY_BASENAME = "acquisition_real_download_2020"
YEAR = 2020
START_DATE = f"{YEAR}-01-01"
END_DATE = f"{YEAR}-12-31"

# Small bbox only for sources whose downloaders/providers actually support it.
USE_SMALL_BBOX_FOR_SUBSETTABLE = True
TEST_BBOX = (110.0, 30.0, 111.0, 31.0)

# If False, methods whose current downloader implementation does not enforce
# temporal slicing are skipped by default.
ALLOW_NON_TEMPORAL_DOWNLOADERS = False
RUN_AUTH_REQUIRED = True

TIMEOUT = 120
MAX_RETRIES = 2
SLEEP_SECONDS = 1.0
CHUNK_SIZE = 1024 * 1024

PREFERRED_VARIABLES: Dict[str, str] = {
    "era5land_hourly": "total_precipitation",
    "imerg_final_hdf5": "precipitation",
    "gleam_monthly_nc": "potential_evaporation",
    "mswep_monthly_nc": "precipitation",
    "mswep_daily_nc": "precipitation",
    "persiann_cdr_erddap": "precipitation",
    "persiann_portal": "precipitation",
    "persiann_ccs_portal": "precipitation",
    "persiann_cdr_portal": "precipitation",
    "persiann_ccs_cdr_v2_portal": "precipitation",
    "pdir_now_portal": "precipitation",
    "persiann_v3_portal": "precipitation",
    "persiann_cdr_v3_portal": "precipitation",
    "cmfd": "precipitation",
    "cmip6_esgf": "pr",
    "cmip5_esgf": "pr",
}

FORMAT_PREFERENCES: Dict[str, str] = {
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

IMERG_USERNAME = os.getenv("IMERG_USERNAME", "lexinlong9@gmail.com")
IMERG_PASSWORD = os.getenv("IMERG_PASSWORD", "lexinlong9@gmail.com")

GLEAM_HOST = os.getenv("GLEAM_HOST", "hydras.ugent.be")
GLEAM_PORT = int(os.getenv("GLEAM_PORT", "2225")) if os.getenv("GLEAM_PORT") else 2225
GLEAM_USERNAME = os.getenv("GLEAM_USERNAME", "gleamuser")
GLEAM_PASSWORD = os.getenv("GLEAM_PASSWORD", "GLEAM4#h-cel_924")
GLEAM_REMOTE_BASE_DIR = os.getenv("GLEAM_REMOTE_BASE_DIR", "")

MSWEP_REMOTE_NAME = os.getenv("MSWEP_REMOTE_NAME", "")
MSWEP_REMOTE_PATH = os.getenv("MSWEP_REMOTE_PATH", "")
RCLONE_BINARY = os.getenv("RCLONE_BINARY", "rclone")

PORTAL_EXPORT_URLS: Dict[str, str] = {
    "persiann_portal": os.getenv("ACQ_EXPORT_URL__PERSIANN_PORTAL", ""),
    "persiann_ccs_portal": os.getenv("ACQ_EXPORT_URL__PERSIANN_CCS_PORTAL", ""),
    "persiann_cdr_portal": os.getenv("ACQ_EXPORT_URL__PERSIANN_CDR_PORTAL", ""),
    "persiann_ccs_cdr_v2_portal": os.getenv("ACQ_EXPORT_URL__PERSIANN_CCS_CDR_V2_PORTAL", ""),
    "pdir_now_portal": os.getenv("ACQ_EXPORT_URL__PDIR_NOW_PORTAL", ""),
    "persiann_v3_portal": os.getenv("ACQ_EXPORT_URL__PERSIANN_V3_PORTAL", ""),
    "persiann_cdr_v3_portal": os.getenv("ACQ_EXPORT_URL__PERSIANN_CDR_V3_PORTAL", ""),
    "cmfd": os.getenv("ACQ_EXPORT_URL__CMFD", ""),
}

ESGF_FACETS: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# Capability policy
# =============================================================================
TEMPORAL_METHODS = {"http_direct", "http_listing", "cds_api", "erddap"}
NON_TEMPORAL_METHODS = {"sftp", "rclone", "portal_export", "esgf", "esgf_wget"}

# Only these methods actually consume bbox in the current codebase.
SUBSETTABLE_METHODS = {"cds_api", "erddap"}

# Optional source-level override. Use this if a future source under a subsettable
# method should still avoid bbox.
FORCE_NO_BBOX_SOURCES = {
    # example: "some_future_source_key"
}

# Optional source-level override. Use this if a future source under a non-
# subsettable method gains explicit spatial export support.
FORCE_BBOX_SOURCES = {
    # example: "some_future_source_key"
}


def bootstrap_acquisition_imports(project_root: Path):
    root = project_root.resolve()
    premise_pkg = root / "premise" / "acquisition"
    acq_pkg = root / "acquisition"

    if premise_pkg.exists():
        sys.path.insert(0, str(root))
        from premise.acquisition.base import DownloadRequest
        from premise.acquisition.catalog import DEFAULT_SOURCES
        from premise.acquisition.router import download_dataset
        return DownloadRequest, DEFAULT_SOURCES, download_dataset

    if acq_pkg.exists():
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

        from premise.acquisition.base import DownloadRequest
        from premise.acquisition.catalog import DEFAULT_SOURCES
        from premise.acquisition.router import download_dataset
        return DownloadRequest, DEFAULT_SOURCES, download_dataset

    raise FileNotFoundError(
        f"Could not find 'premise/acquisition' or 'acquisition' under import root: {root}"
    )


def unique_sources(default_sources) -> List[Any]:
    seen = set()
    out = []
    for src in default_sources:
        key = src.key.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(src)
    return out


def choose_variable(src) -> str:
    return PREFERRED_VARIABLES.get(src.key, src.variables[0])


def source_supports_bbox(src) -> bool:
    key = src.key.lower().strip()
    method = src.method.lower().strip()
    if key in FORCE_BBOX_SOURCES:
        return True
    if key in FORCE_NO_BBOX_SOURCES:
        return False
    return method in SUBSETTABLE_METHODS


def choose_bbox(src) -> Optional[Tuple[float, float, float, float]]:
    if not USE_SMALL_BBOX_FOR_SUBSETTABLE:
        return None
    if source_supports_bbox(src):
        return TEST_BBOX
    return None


def build_request(DownloadRequest, src, outdir: Path):
    var = choose_variable(src)
    return DownloadRequest(
        source_key=src.key,
        variables=(var,),
        start_date=START_DATE,
        end_date=END_DATE,
        bbox=choose_bbox(src),
        target_dir=str(outdir),
        notes="real download test for 2020",
        frequency=src.temporal_resolution,
        format_preference=FORMAT_PREFERENCES.get(src.key),
    )


def build_downloader_kwargs(src) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    method = src.method.lower().strip()

    if src.auth_required and not RUN_AUTH_REQUIRED:
        return None, "auth_required_but_RUN_AUTH_REQUIRED_is_False"

    if method in NON_TEMPORAL_METHODS and not ALLOW_NON_TEMPORAL_DOWNLOADERS:
        return None, "current_downloader_does_not_enforce_2020_time_filter"

    kwargs: Dict[str, Any] = {
        "timeout": TIMEOUT,
        "max_retries": MAX_RETRIES,
        "sleep_seconds": SLEEP_SECONDS,
    }

    if method in {"http_direct", "http_listing", "erddap", "portal_export"}:
        kwargs["chunk_size"] = CHUNK_SIZE

    if method == "http_listing":
        if src.key == "imerg_final_hdf5":
            if not (IMERG_USERNAME and IMERG_PASSWORD):
                return None, "missing_IMERG_USERNAME_or_IMERG_PASSWORD"
            kwargs["username"] = IMERG_USERNAME
            kwargs["password"] = IMERG_PASSWORD
        return kwargs, None

    if method in {"http_direct", "erddap", "cds_api"}:
        return kwargs, None

    if method == "sftp":
        if not (GLEAM_HOST and GLEAM_USERNAME and GLEAM_PASSWORD):
            return None, "missing_GLEAM_SFTP_credentials"
        kwargs.update({
            "host": GLEAM_HOST,
            "port": GLEAM_PORT,
            "username": GLEAM_USERNAME,
            "password": GLEAM_PASSWORD,
        })
        if GLEAM_REMOTE_BASE_DIR:
            kwargs["remote_base_dir"] = GLEAM_REMOTE_BASE_DIR
        return kwargs, None

    if method == "rclone":
        if not (MSWEP_REMOTE_NAME and MSWEP_REMOTE_PATH):
            return None, "missing_MSWEP_REMOTE_NAME_or_MSWEP_REMOTE_PATH"
        kwargs.update({
            "remote_name": MSWEP_REMOTE_NAME,
            "remote_path": MSWEP_REMOTE_PATH,
            "rclone_binary": RCLONE_BINARY,
        })
        return kwargs, None

    if method == "portal_export":
        export_url = PORTAL_EXPORT_URLS.get(src.key, "")
        if not export_url:
            return None, "missing_portal_export_url"
        kwargs["export_url"] = export_url
        return kwargs, None

    if method == "esgf":
        facets = ESGF_FACETS.get(src.key)
        if not facets:
            return None, "missing_ESGF_FACETS_for_search"
        kwargs["facets"] = facets
        return kwargs, None

    if method == "esgf_wget":
        return None, "catalog_currently_uses_esgf_not_esgf_wget"

    return None, f"unsupported_method_{method}"


def summarize_files(paths: List[Path]) -> Dict[str, Any]:
    existing = [Path(p) for p in paths if Path(p).exists()]
    total_size = sum(p.stat().st_size for p in existing)
    return {
        "file_count": len(existing),
        "total_size_bytes": total_size,
        "files": [str(p) for p in existing],
    }


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


if __name__ == "__main__":
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    DownloadRequest, DEFAULT_SOURCES, download_dataset = bootstrap_acquisition_imports(PROJECT_ROOT)
    sources = unique_sources(DEFAULT_SOURCES)

    dataset_rows: List[Dict[str, Any]] = []

    print("=" * 88)
    print("Real acquisition download test for 2020")
    print(f"SCRIPT_DIR    : {SCRIPT_DIR}")
    print(f"PROJECT_ROOT  : {PROJECT_ROOT}")
    print(f"DOWNLOAD_ROOT : {DOWNLOAD_ROOT}")
    print(f"DATE RANGE    : {START_DATE} -> {END_DATE}")
    print(f"TEST_BBOX     : {TEST_BBOX if USE_SMALL_BBOX_FOR_SUBSETTABLE else 'disabled'}")
    print(f"ALLOW_NON_TEMPORAL_DOWNLOADERS : {ALLOW_NON_TEMPORAL_DOWNLOADERS}")
    print("=" * 88)

    for src in sources:
        src_dir = DOWNLOAD_ROOT / src.key
        src_dir.mkdir(parents=True, exist_ok=True)

        bbox = choose_bbox(src)
        row: Dict[str, Any] = {
            "source_key": src.key,
            "title": src.title,
            "method": src.method,
            "auth_required": src.auth_required,
            "variable": choose_variable(src),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "bbox": bbox,
            "bbox_mode": "subset" if bbox is not None else "none",
            "target_dir": str(src_dir),
            "status": "pending",
            "message": "",
            "file_count": 0,
            "total_size_bytes": 0,
        }

        downloader_kwargs, skip_reason = build_downloader_kwargs(src)
        if skip_reason is not None:
            row["status"] = "skipped"
            row["message"] = skip_reason
            dataset_rows.append(row)
            print(f"[SKIP] {src.key}: {skip_reason}")
            continue

        request = build_request(DownloadRequest, src, src_dir)

        try:
            print(f"[RUN ] {src.key} ({src.method}) bbox={'yes' if bbox is not None else 'no'}")
            outputs = download_dataset(request, **downloader_kwargs)
            stats = summarize_files(outputs)
            row["status"] = "downloaded" if stats["file_count"] > 0 else "no_files_returned"
            row["message"] = "ok" if stats["file_count"] > 0 else "downloader_returned_no_files"
            row["file_count"] = stats["file_count"]
            row["total_size_bytes"] = stats["total_size_bytes"]
            row["files_json"] = json.dumps(stats["files"], ensure_ascii=False)
            print(f"[ OK ] {src.key}: {stats['file_count']} file(s), {stats['total_size_bytes']} bytes")
        except Exception as e:
            row["status"] = "failed"
            row["message"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()
            print(f"[FAIL] {src.key}: {type(e).__name__}: {e}")

        dataset_rows.append(row)

    summary = {
        "script_dir": str(SCRIPT_DIR),
        "project_root": str(PROJECT_ROOT),
        "download_root": str(DOWNLOAD_ROOT),
        "year": YEAR,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "small_bbox": TEST_BBOX if USE_SMALL_BBOX_FOR_SUBSETTABLE else None,
        "allow_non_temporal_downloaders": ALLOW_NON_TEMPORAL_DOWNLOADERS,
        "run_auth_required": RUN_AUTH_REQUIRED,
        "dataset_count": len(dataset_rows),
        "downloaded": sum(1 for r in dataset_rows if r["status"] == "downloaded"),
        "skipped": sum(1 for r in dataset_rows if r["status"] == "skipped"),
        "failed": sum(1 for r in dataset_rows if r["status"] == "failed"),
        "no_files_returned": sum(1 for r in dataset_rows if r["status"] == "no_files_returned"),
    }

    summary_json = DOWNLOAD_ROOT / f"{SUMMARY_BASENAME}_summary.json"
    datasets_json = DOWNLOAD_ROOT / f"{SUMMARY_BASENAME}_datasets.json"
    datasets_csv = DOWNLOAD_ROOT / f"{SUMMARY_BASENAME}_datasets.csv"

    write_json(summary_json, summary)
    write_json(datasets_json, dataset_rows)
    write_csv(datasets_csv, dataset_rows)

    print("-" * 88)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {summary_json}")
    print(f"Saved: {datasets_json}")
    print(f"Saved: {datasets_csv}")
