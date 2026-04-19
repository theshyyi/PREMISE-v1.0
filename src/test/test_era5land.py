import json
from pathlib import Path

import pandas as pd

from premise.acquisition import DownloadRequest, download_dataset, list_sources


def test_era5land_monthly_downloader_smoke():
    candidates = [s for s in list_sources(variable="2m_temperature") if s.key == "era5land_monthly"]
    assert len(candidates) == 1, "ERA5-Land monthly source was not found in catalog."

    outdir = Path(r"I:\论文手稿（ESR）\论文EMS\PREMISE-v1.0\tests_output\era5land_monthly_smoke")

    req = DownloadRequest(
        source_key="era5land_monthly",
        variables=("2m_temperature",),
        start_date="2020-01-01",
        end_date="2020-03-31",
        target_dir=str(outdir),
        notes="smoke test for ERA5-Land monthly downloader",
        frequency="monthly",
        format_preference="netcdf",
    )

    files = download_dataset(req, timeout=300, max_retries=1)

    assert isinstance(files, list)
    assert len(files) >= 1
    assert all(Path(f).exists() for f in files)

    manifest_path = outdir / "acquisition_manifest.json"
    summary_path = outdir / "acquisition_summary.csv"
    file_records_path = outdir / "acquisition_file_records.csv"
    event_log_path = outdir / "acquisition_events.log"

    assert manifest_path.exists()
    assert summary_path.exists()
    assert file_records_path.exists()
    assert event_log_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["source_key"] == "era5land_monthly"
    assert manifest["provider_name"] == "ERA5-Land"
    assert manifest["requested_file_count"] >= 1
    assert manifest["success_count"] + manifest["skipped_existing_count"] >= 1

    df_summary = pd.read_csv(summary_path)
    assert len(df_summary) == 1
    assert df_summary.iloc[0]["source_key"] == "era5land_monthly"

    df_records = pd.read_csv(file_records_path)
    assert len(df_records) >= 1
    assert "size_mb" in df_records.columns
    assert "elapsed_seconds" in df_records.columns


if __name__ == "__main__":
    test_era5land_monthly_downloader_smoke()
    print("ERA5-Land monthly smoke test finished.")