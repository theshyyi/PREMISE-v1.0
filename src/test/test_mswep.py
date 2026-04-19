import json
import os
from pathlib import Path

import pandas as pd

from premise.acquisition import DownloadRequest, download_dataset, list_sources


def test_mswep_monthly_rclone_smoke():
    remote_name = os.environ.get('MSWEP_RCLONE_REMOTE')
    remote_path = os.environ.get('MSWEP_RCLONE_PATH')
    subdir = os.environ.get('MSWEP_RCLONE_SUBDIR', 'Monthly')
    if not remote_name or not remote_path:
        print('SKIPPED: set MSWEP_RCLONE_REMOTE and MSWEP_RCLONE_PATH to run this test.')
        return
    candidates = [s for s in list_sources(variable='precipitation') if s.key == 'mswep_monthly_nc']
    assert len(candidates) == 1
    outdir = Path(r"I:\论文手稿（ESR）\论文EMS\PREMISE-v1.0\tests_output\mswep_monthly_nc_smoke")
    req = DownloadRequest(source_key='mswep_monthly_nc', variables=('precipitation',), start_date='2020-01-01', end_date='2020-03-31', target_dir=str(outdir), notes='smoke test for MSWEP monthly downloader', frequency='monthly', format_preference='NetCDF')
    files = download_dataset(req, timeout=600, max_retries=1, remote_name=remote_name, remote_path=remote_path, subdir=subdir, dry_run=False)
    assert isinstance(files, list) and len(files) >= 1 and all(Path(f).exists() for f in files)
    with open(outdir / 'acquisition_manifest.json', 'r', encoding='utf-8') as f: manifest = json.load(f)
    assert manifest['source_key'] == 'mswep_monthly_nc'
    df_summary = pd.read_csv(outdir / 'acquisition_summary.csv'); assert len(df_summary) == 1


if __name__ == '__main__':
    test_mswep_monthly_rclone_smoke(); print('MSWEP monthly rclone smoke test finished.')
