import json
import os
from pathlib import Path

import pandas as pd

from premise.acquisition import DownloadRequest, download_dataset, list_sources


def test_persiann_family_portal_smoke():
    export_url = os.environ.get('PERSIANN_PORTAL_EXPORT_URL')
    if not export_url or not export_url.startswith('http'):
        print('SKIPPED: set PERSIANN_PORTAL_EXPORT_URL to run this portal smoke test.')
        return
    candidates = [s for s in list_sources(variable='precipitation') if s.key == 'persiann_v3_portal']
    assert len(candidates) == 1
    outdir = Path(r"I:\论文手稿（ESR）\论文EMS\PREMISE-v1.0\tests_output\persiann_v3_portal_smoke")
    req = DownloadRequest(source_key='persiann_v3_portal', variables=('precipitation',), start_date='2020-01-01', end_date='2020-01-03', target_dir=str(outdir), notes='smoke test for PERSIANN family portal downloader', frequency='daily', format_preference='NetCDF')
    files = download_dataset(req, timeout=120, max_retries=2, export_url=export_url)
    assert isinstance(files, list) and len(files) >= 1 and all(Path(f).exists() for f in files)
    with open(outdir / 'acquisition_manifest.json', 'r', encoding='utf-8') as f: manifest = json.load(f)
    assert manifest['source_key'] == 'persiann_v3_portal'
    df_summary = pd.read_csv(outdir / 'acquisition_summary.csv'); assert len(df_summary) == 1


if __name__ == '__main__':
    test_persiann_family_portal_smoke(); print('PERSIANN family portal smoke test finished.')
