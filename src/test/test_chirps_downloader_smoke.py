import json
from pathlib import Path

import pandas as pd

from premise.acquisition import DownloadRequest, download_dataset, list_sources


def test_chirps_downloader_monthly_smoke():
    candidates = [s for s in list_sources(variable="precipitation") if s.key == "chirps_monthly_nc"]
    assert len(candidates) == 1, "CHIRPS source was not found in catalog."
    outdir = Path(r"E:\论文手稿（ESR）\论文EMS\PREMISE-v1.0\tests_output\SECTION4")
    req = DownloadRequest(source_key="chirps_monthly_nc", variables=("precipitation",), start_date="2020-01-01", end_date="2025-01-01", target_dir=str(outdir), notes="smoke test for CHIRPS monthly downloader", frequency="monthly", format_preference="NetCDF")

    files = download_dataset(req, timeout=120, max_retries=2)
    assert isinstance(files, list) and len(files) >= 1 and all(Path(f).exists() for f in files)
    manifest_path = outdir / "acquisition_manifest.json"; summary_path = outdir / "acquisition_summary.csv"; file_records_path = outdir / "acquisition_file_records.csv"; event_log_path = outdir / "acquisition_events.log"
    assert manifest_path.exists() and summary_path.exists() and file_records_path.exists() and event_log_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f: manifest = json.load(f)
    assert manifest["source_key"] == "chirps_monthly_nc"
    df_summary = pd.read_csv(summary_path); assert len(df_summary) == 1


if __name__ == "__main__":
    test_chirps_downloader_monthly_smoke(); print("CHIRPS monthly smoke test finished.")
