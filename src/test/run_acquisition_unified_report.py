from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import os
import re
import sys
import types
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pytest


# =============================================================================
# Import helper
# =============================================================================

def _load_acquisition_package():
    """
    Load the acquisition package in a way that works for either:
      1) an installed package: premise.acquisition
      2) a local checkout where only ./acquisition exists
    """
    try:
        return importlib.import_module("premise.acquisition")
    except Exception:
        pass

    root_candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
    ]
    for root in root_candidates:
        acq_dir = root / "acquisition"
        if not acq_dir.exists() or not (acq_dir / "__init__.py").exists():
            continue

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        premise_pkg = sys.modules.get("premise")
        if premise_pkg is None:
            premise_pkg = types.ModuleType("premise")
            premise_pkg.__path__ = [str(root)]
            sys.modules["premise"] = premise_pkg

        spec = importlib.util.spec_from_file_location(
            "premise.acquisition",
            acq_dir / "__init__.py",
            submodule_search_locations=[str(acq_dir)],
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules["premise.acquisition"] = module
        setattr(premise_pkg, "acquisition", module)
        spec.loader.exec_module(module)
        return module

    raise ImportError(
        "Unable to import acquisition package. Expected either an installed "
        "'premise.acquisition' or a local './acquisition' directory."
    )


ACQ = _load_acquisition_package()
CATALOG = importlib.import_module(f"{ACQ.__name__}.catalog")
BASE = importlib.import_module(f"{ACQ.__name__}.base")
ROUTER = importlib.import_module(f"{ACQ.__name__}.router")
REGISTRY = importlib.import_module(f"{ACQ.__name__}.registry")

SOURCES = tuple(CATALOG.list_sources())
SOURCE_KEYS = {src.key for src in SOURCES}
SOURCE_BY_KEY = {src.key: src for src in SOURCES}


# =============================================================================
# Helpers
# =============================================================================

def _sample_request(source, target_dir: Path):
    variable = source.variables[0] if source.variables else "precipitation"
    fmt_pref = None
    if source.method == "cds_api":
        fmt_pref = source.params.get("default_data_format", "grib")

    return BASE.DownloadRequest(
        source_key=source.key,
        variables=(variable,),
        start_date="2001-01-01",
        end_date="2001-01-03",
        bbox=(100.0, 20.0, 101.0, 21.0),
        target_dir=str(target_dir),
        notes="unified test",
        format_preference=fmt_pref,
    )


def _build_http_direct_probe_url(source) -> str:
    cls = REGISTRY.get_downloader_class_by_method("http_direct")
    params = source.params
    time_expansion = params.get("time_expansion", "year")
    ctxs = cls._expand_contexts(  # noqa: SLF001
        start_date="2001-01-01",
        end_date="2001-01-03",
        time_expansion=time_expansion,
        params=params,
    )
    ctx = ctxs[0]
    filename = params["filename_pattern"].format(**ctx)
    subdir = params.get("directory_pattern", "")
    if subdir:
        subdir = subdir.format(**ctx)
    return urljoin(params["base_url"], f"{subdir}{filename}")


def _build_http_listing_probe_url(source) -> str:
    cls = REGISTRY.get_downloader_class_by_method("http_listing")
    params = source.params
    time_expansion = params.get("time_expansion", "date")
    ctxs = cls._expand_contexts(  # noqa: SLF001
        start_date="2001-01-01",
        end_date="2001-01-03",
        time_expansion=time_expansion,
    )
    ctx = ctxs[0]
    directory_pattern = params.get("directory_pattern", "")
    if directory_pattern:
        return urljoin(params["base_url"], directory_pattern.format(**ctx))
    return params["base_url"]


def _public_live_sources() -> Iterable:
    supported = {"http_direct", "http_listing", "ftp", "erddap"}
    for src in SOURCES:
        if src.auth_required:
            continue
        if src.method not in supported:
            continue
        yield src


def _extract_source_key(nodeid: str) -> str | None:
    m = re.search(r"\[([^\]]+)\]$", nodeid)
    if not m:
        return None
    key = m.group(1)
    return key if key in SOURCE_KEYS else None


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _parse_junit_to_rows(xml_path: Path) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    rows: list[dict] = []
    for testcase in root.iter("testcase"):
        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "")
        nodeid = f"{classname}::{name}" if classname else name
        duration = float(testcase.attrib.get("time", 0.0) or 0.0)

        outcome = "passed"
        message = ""
        if testcase.find("failure") is not None:
            outcome = "failed"
            failure = testcase.find("failure")
            message = (failure.text or "").strip() or failure.attrib.get("message", "")
        elif testcase.find("error") is not None:
            outcome = "failed"
            err = testcase.find("error")
            message = (err.text or "").strip() or err.attrib.get("message", "")
        elif testcase.find("skipped") is not None:
            outcome = "skipped"
            sk = testcase.find("skipped")
            message = sk.attrib.get("message", "") or (sk.text or "").strip()

        rows.append({
            "nodeid": nodeid,
            "test_name": name,
            "outcome": outcome,
            "duration_sec": round(duration, 6),
            "source_key": _extract_source_key(name) or _extract_source_key(nodeid),
            "message": message,
        })
    return rows


def _build_dataset_rows(test_rows: list[dict]) -> list[dict]:
    rows_by_source: dict[str, list[dict]] = defaultdict(list)
    for row in test_rows:
        if row["source_key"]:
            rows_by_source[row["source_key"]].append(row)

    dataset_rows: list[dict] = []
    for src in SOURCES:
        rows = rows_by_source.get(src.key, [])
        counts = Counter(r["outcome"] for r in rows)
        failed_examples = [r["test_name"] for r in rows if r["outcome"] == "failed"][:3]
        skipped_examples = [r["test_name"] for r in rows if r["outcome"] == "skipped"][:3]

        if counts["failed"] > 0:
            status = "failed"
        elif counts["passed"] > 0:
            status = "passed"
        elif counts["skipped"] > 0:
            status = "skipped"
        else:
            status = "not_covered"

        dataset_rows.append({
            "source_key": src.key,
            "title": src.title,
            "provider": src.provider,
            "method": src.method,
            "auth_required": src.auth_required,
            "official_url": src.official_url,
            "variables": ", ".join(src.variables),
            "total_checks": len(rows),
            "passed": counts["passed"],
            "failed": counts["failed"],
            "skipped": counts["skipped"],
            "status": status,
            "failed_examples": " | ".join(failed_examples),
            "skipped_examples": " | ".join(skipped_examples),
        })
    return dataset_rows


def _write_reports(report_dir: Path, basename: str, exit_code: int, test_rows: list[dict]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = _build_dataset_rows(test_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "exit_code": int(exit_code),
        "totals": {
            "tests_total": len(test_rows),
            "passed": sum(r["outcome"] == "passed" for r in test_rows),
            "failed": sum(r["outcome"] == "failed" for r in test_rows),
            "skipped": sum(r["outcome"] == "skipped" for r in test_rows),
            "sources_total": len(SOURCES),
            "sources_failed": sum(r["failed"] > 0 for r in dataset_rows),
            "sources_all_passed": sum(r["status"] == "passed" for r in dataset_rows),
        },
        "files": {
            "summary_json": f"{basename}_summary.json",
            "tests_csv": f"{basename}_tests.csv",
            "datasets_csv": f"{basename}_datasets.csv",
            "datasets_json": f"{basename}_datasets.json",
            "junit_xml": f"{basename}_junit.xml",
        },
    }

    (report_dir / f"{basename}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / f"{basename}_datasets.json").write_text(
        json.dumps(dataset_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        report_dir / f"{basename}_tests.csv",
        test_rows,
        ["nodeid", "test_name", "outcome", "duration_sec", "source_key", "message"],
    )
    _write_csv(
        report_dir / f"{basename}_datasets.csv",
        dataset_rows,
        [
            "source_key", "title", "provider", "method", "auth_required", "official_url",
            "variables", "total_checks", "passed", "failed", "skipped", "status",
            "failed_examples", "skipped_examples",
        ],
    )


# =============================================================================
# Tests
# =============================================================================

def test_catalog_is_not_empty():
    assert SOURCES, "No data sources found in catalog."


def test_source_keys_are_unique():
    keys = [src.key for src in SOURCES]
    assert len(keys) == len(set(keys)), "Duplicate source keys detected in catalog."


def test_aliases_do_not_collide_with_other_source_keys():
    keys = {src.key for src in SOURCES}
    for src in SOURCES:
        for alias in src.aliases:
            assert alias not in (keys - {src.key}), (
                f"Alias '{alias}' from source '{src.key}' collides with another source key."
            )


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.key)
def test_all_sources_have_required_metadata(source):
    assert source.key.strip()
    assert source.title.strip()
    assert source.provider.strip()
    assert source.method.strip()
    assert isinstance(source.variables, tuple)
    assert len(source.variables) >= 1
    assert source.temporal_resolution.strip()
    assert source.spatial_resolution.strip()
    assert source.official_url.strip()
    assert source.access_mode.strip()
    assert isinstance(source.params, dict)


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.key)
def test_all_source_methods_are_registered(source):
    cls = REGISTRY.get_downloader_class_by_method(source.method)
    assert cls is not None


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.key)
def test_all_sources_can_instantiate_downloader(source):
    cls = REGISTRY.get_downloader_class_by_method(source.method)
    downloader = cls(source_config=source)
    assert downloader.source_config.key == source.key


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.key)
def test_router_dispatch_works_for_every_source(monkeypatch, tmp_path, source):
    cls = REGISTRY.get_downloader_class_by_method(source.method)

    def _fake_download(self, request):
        outdir = Path(request.target_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"{request.source_key}.ok"
        out.write_text("ok", encoding="utf-8")
        return [out]

    monkeypatch.setattr(cls, "download", _fake_download, raising=True)
    req = _sample_request(source, tmp_path / source.key)
    files = ROUTER.download_dataset(req)

    assert len(files) == 1
    assert files[0].exists()
    assert files[0].name == f"{source.key}.ok"


HTTP_DIRECT_SOURCES = tuple(src for src in SOURCES if src.method == "http_direct")
HTTP_LISTING_SOURCES = tuple(src for src in SOURCES if src.method == "http_listing")
FTP_SOURCES = tuple(src for src in SOURCES if src.method == "ftp")
ERDDAP_SOURCES = tuple(src for src in SOURCES if src.method == "erddap")
CDS_SOURCES = tuple(src for src in SOURCES if src.method == "cds_api")
SFTP_SOURCES = tuple(src for src in SOURCES if src.method == "sftp")
RCLONE_SOURCES = tuple(src for src in SOURCES if src.method == "rclone")
EARTHDATA_SOURCES = tuple(src for src in SOURCES if src.method == "earthdata_cmr")
ESGF_SOURCES = tuple(src for src in SOURCES if src.method == "esgf")
PORTAL_SOURCES = tuple(src for src in SOURCES if src.method == "portal_export")
ESGF_WGET_SOURCES = tuple(src for src in SOURCES if src.method == "esgf_wget")
WGET_SOURCES = tuple(src for src in SOURCES if src.method == "wget")


@pytest.mark.parametrize("source", HTTP_DIRECT_SOURCES, ids=lambda s: s.key)
def test_http_direct_sources_have_minimum_contract(source):
    assert "base_url" in source.params
    assert "filename_pattern" in source.params
    assert source.params.get("time_expansion", "year") in {"year", "date", "none", "static", "single", "decade"}


@pytest.mark.parametrize("source", HTTP_LISTING_SOURCES, ids=lambda s: s.key)
def test_http_listing_sources_have_minimum_contract(source):
    assert "base_url" in source.params
    assert source.params.get("time_expansion", "date") in {"year", "date", "none", "static", "single"}


@pytest.mark.parametrize("source", FTP_SOURCES, ids=lambda s: s.key)
def test_ftp_sources_have_minimum_contract(source):
    assert "host" in source.params
    assert "remote_base_dir" in source.params
    assert ("filename_pattern" in source.params) or ("file_regex" in source.params)


@pytest.mark.parametrize("source", ERDDAP_SOURCES, ids=lambda s: s.key)
def test_erddap_sources_have_minimum_contract(source):
    assert "base_url" in source.params
    assert "query_variable" in source.params


@pytest.mark.parametrize("source", CDS_SOURCES, ids=lambda s: s.key)
def test_cds_sources_have_minimum_contract(source):
    assert "dataset_name" in source.params
    assert source.params.get("time_mode")


@pytest.mark.parametrize("source", SFTP_SOURCES, ids=lambda s: s.key)
def test_sftp_sources_have_minimum_contract(source):
    assert "default_port" in source.params


@pytest.mark.parametrize("source", RCLONE_SOURCES, ids=lambda s: s.key)
def test_rclone_sources_have_minimum_contract(source):
    assert "default_subdir" in source.params


@pytest.mark.parametrize("source", EARTHDATA_SOURCES, ids=lambda s: s.key)
def test_earthdata_sources_have_minimum_contract(source):
    assert source.params.get("short_name") or source.params.get("concept_id")
    assert source.params.get("provider") or source.params.get("cmr_granules_url")


@pytest.mark.parametrize("source", ESGF_SOURCES, ids=lambda s: s.key)
def test_esgf_sources_have_minimum_contract(source):
    assert "project" in source.params
    assert "default_search_node" in source.params


@pytest.mark.parametrize("source", ESGF_WGET_SOURCES, ids=lambda s: s.key)
def test_esgf_wget_sources_have_minimum_contract(source):
    assert source.params is not None


@pytest.mark.parametrize("source", WGET_SOURCES, ids=lambda s: s.key)
def test_wget_sources_have_minimum_contract(source):
    assert source.params is not None


@pytest.mark.parametrize("source", PORTAL_SOURCES, ids=lambda s: s.key)
def test_portal_sources_are_portal_export(source):
    assert source.method == "portal_export"


RUN_LIVE = os.getenv("RUN_ACQUISITION_LIVE", "0") == "1"


@pytest.mark.skipif(not RUN_LIVE, reason="Set RUN_ACQUISITION_LIVE=1 to enable live endpoint probes.")
@pytest.mark.parametrize("source", tuple(_public_live_sources()), ids=lambda s: s.key)
def test_public_endpoint_reachable(source):
    timeout = 20

    if source.method == "http_direct":
        import requests
        url = _build_http_direct_probe_url(source)
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
            assert resp.status_code < 400, f"Probe failed for {source.key}: {url} -> {resp.status_code}"
        return

    if source.method == "http_listing":
        import requests
        url = _build_http_listing_probe_url(source)
        with requests.get(url, timeout=timeout, allow_redirects=True) as resp:
            assert resp.status_code < 400, f"Probe failed for {source.key}: {url} -> {resp.status_code}"
        return

    if source.method == "ftp":
        import ftplib
        host = source.params["host"]
        remote_dir = source.params["remote_base_dir"]
        with ftplib.FTP(host=host, timeout=timeout) as ftp:
            ftp.login()
            ftp.cwd(remote_dir)
            ftp.nlst()
        return

    if source.method == "erddap":
        import requests
        base_url = source.params["base_url"]
        probe_url = base_url[:-3] + ".html" if base_url.endswith(".nc") else base_url
        with requests.get(probe_url, timeout=timeout, allow_redirects=True) as resp:
            assert resp.status_code < 400, f"Probe failed for {source.key}: {probe_url} -> {resp.status_code}"
        return

    pytest.skip(f"No live probe implemented for method={source.method}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    report_dir = Path(os.getenv("ACQ_REPORT_DIR", "test_reports"))
    basename = os.getenv("ACQ_REPORT_BASENAME", "acquisition_unified")
    report_dir.mkdir(parents=True, exist_ok=True)
    junit_xml = report_dir / f"{basename}_junit.xml"

    exit_code = pytest.main([__file__, "-q", f"--junitxml={junit_xml}"])
    test_rows = _parse_junit_to_rows(junit_xml) if junit_xml.exists() else []
    _write_reports(report_dir, basename, exit_code, test_rows)

    print(f"[acq-report] reports written to: {report_dir.resolve()}")
    print(f"[acq-report] summary JSON: {report_dir / (basename + '_summary.json')}")
    print(f"[acq-report] dataset CSV : {report_dir / (basename + '_datasets.csv')}")
    print(f"[acq-report] test CSV    : {report_dir / (basename + '_tests.csv')}")
    raise SystemExit(exit_code)
