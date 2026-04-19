import json
from pathlib import Path

import pandas as pd

from premise.acquisition import DownloadRequest, download_dataset, list_sources


def test_era5land_hourly_downloader_smoke():
    candidates = [s for s in list_sources(variable="2m_temperature") if s.key == "era5land_hourly"]
    assert len(candidates) == 1
    outdir = Path(r"I:\论文手稿（ESR）\论文EMS\PREMISE-v1.0\tests_output\era5land_hourly_smoke")
    req = DownloadRequest(source_key="era5land_hourly", variables=("2m_temperature",), start_date="2020-01-01", end_date="2020-01-01", target_dir=str(outdir), notes="smoke test for ERA5-Land hourly downloader", frequency="hourly", format_preference="grib")
    files = download_dataset(req, timeout=300, max_retries=1)
    assert isinstance(files, list) and len(files) >= 1 and all(Path(f).exists() for f in files)
    with open(outdir / 'acquisition_manifest.json', 'r', encoding='utf-8') as f: manifest = json.load(f)
    assert manifest['source_key'] == 'era5land_hourly'
    df_summary = pd.read_csv(outdir / 'acquisition_summary.csv'); assert len(df_summary) == 1


if __name__ == '__main__':
    test_era5land_hourly_downloader_smoke(); print('ERA5-Land hourly smoke test finished.')
