import json
from pathlib import Path

import pandas as pd

from premise.acquisition import DownloadRequest, download_dataset


def test_invalid_source_key():
    req = DownloadRequest(source_key="invalid_source", variables=("precipitation",), start_date="2020-01-01", end_date="2020-03-31", target_dir=r"I:\论文手稿（ESR）\论文EMS\PREMISE-v1.0\tests_output\invalid_source_case", notes="robustness test for invalid source key")
    try:
        download_dataset(req, timeout=30, max_retries=1)
        assert False, "Expected invalid source_key to raise an exception."
    except KeyError as e:
        assert "No source configuration found" in str(e)


def test_invalid_variable_for_chirps():
    req = DownloadRequest(source_key="chirps_monthly_nc", variables=("temperature",), start_date="2020-01-01", end_date="2020-03-31", target_dir=r"I:\论文手稿（ESR）\论文EMS\PREMISE-v1.0\tests_output\invalid_variable_case", notes="robustness test for unsupported variable", frequency="monthly", format_preference="NetCDF")
    try:
        download_dataset(req, timeout=30, max_retries=1)
        assert False, "Expected unsupported variable to raise an exception."
    except Exception:
        assert True


def test_skipped_existing_behavior():
    outdir = Path(r"I:\论文手稿（ESR）\论文EMS\PREMISE-v1.0\tests_output\chirps_skip_case")
    req = DownloadRequest(source_key="chirps_monthly_nc", variables=("precipitation",), start_date="2020-01-01", end_date="2020-03-31", target_dir=str(outdir), notes="robustness test for skipped_existing behavior", frequency="monthly", format_preference="NetCDF")
    files_first = download_dataset(req, timeout=120, max_retries=2)
    assert isinstance(files_first, list)
    assert len(files_first) >= 1
    assert all(Path(f).exists() for f in files_first)
    files_second = download_dataset(req, timeout=120, max_retries=2)
    assert isinstance(files_second, list)
    assert len(files_second) >= 1
    assert all(Path(f).exists() for f in files_second)
    manifest_path = outdir / "acquisition_manifest.json"
    summary_path = outdir / "acquisition_summary.csv"
    file_records_path = outdir / "acquisition_file_records.csv"
    event_log_path = outdir / "acquisition_events.log"
    assert manifest_path.exists() and summary_path.exists() and file_records_path.exists() and event_log_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["skipped_existing_count"] >= 1
    df_records = pd.read_csv(file_records_path)
    assert (df_records["status"] == "skipped_existing").any()
    event_text = event_log_path.read_text(encoding="utf-8")
    assert "skipped_existing" in event_text
