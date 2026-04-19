import json
from pathlib import Path

import pandas as pd

from premise.acquisition import DownloadRequest, download_dataset, list_sources


def test_persiann_cdr_erddap_smoke():
    candidates = [s for s in list_sources(variable='precipitation') if s.key == 'persiann_cdr_erddap']
    assert len(candidates) == 1
    outdir = Path(r"I:\论文手稿（ESR）\论文EMS\PREMISE-v1.0\tests_output\persiann_cdr_erddap_smoke")
    req = DownloadRequest(source_key='persiann_cdr_erddap', variables=('precipitation',), start_date='2020-01-01', end_date='2020-01-03', target_dir=str(outdir), notes='smoke test for PERSIANN-CDR ERDDAP downloader', frequency='daily', format_preference='NetCDF')
    files = download_dataset(req, timeout=120, max_retries=2)
    assert isinstance(files, list) and len(files) >= 1 and all(Path(f).exists() for f in files)
    with open(outdir / 'acquisition_manifest.json', 'r', encoding='utf-8') as f: manifest = json.load(f)
    assert manifest['source_key'] == 'persiann_cdr_erddap'
    df_summary = pd.read_csv(outdir / 'acquisition_summary.csv'); assert len(df_summary) == 1


if __name__ == '__main__':
    test_persiann_cdr_erddap_smoke(); print('PERSIANN-CDR ERDDAP smoke test finished.')
