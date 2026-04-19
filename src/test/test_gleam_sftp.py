import json
import os
from pathlib import Path

import pandas as pd

from premise.acquisition import DownloadRequest, download_dataset, list_sources


def test_gleam_sftp_downloader_smoke():
    host = 'hydras.ugent.be'
        # os.environ.get('hydras.ugent.be')
    username = 'gleamuser'
        # os.environ.get('gleamuser')

    password = 'GLEAM4#h-cel_924'
        # os.environ.get('GLEAM4#h-cel_924')
    remote_base_dir = os.environ.get('GLEAM_REMOTE_BASE_DIR', './data/v4.2b/monthly/Ep')
    if not all([host, username, password]):
        print('SKIPPED: set GLEAM_SFTP_HOST, GLEAM_SFTP_USERNAME, and GLEAM_SFTP_PASSWORD to run this test.')
        return
    candidates = [s for s in list_sources(variable='potential_evaporation') if s.key == 'gleam_monthly_nc']
    assert len(candidates) == 1
    outdir = Path(r"I:\论文手稿（ESR）\论文EMS\PREMISE-v1.0\tests_output\gleam_monthly_nc_smoke")
    req = DownloadRequest(source_key='gleam_monthly_nc', variables=('potential_evaporation',), start_date='2020-01-01', end_date='2020-03-31', target_dir=str(outdir), notes='smoke test for GLEAM monthly NetCDF downloader', frequency='monthly', format_preference='NetCDF')
    files = download_dataset(req, timeout=120, max_retries=1, file_limit=1, host=host, port=2225, username=username, password=password, remote_base_dir=remote_base_dir)
    assert isinstance(files, list) and len(files) >= 1 and all(Path(f).exists() for f in files)
    with open(outdir / 'acquisition_manifest.json', 'r', encoding='utf-8') as f: manifest = json.load(f)
    assert manifest['source_key'] == 'gleam_monthly_nc'
    df_summary = pd.read_csv(outdir / 'acquisition_summary.csv'); assert len(df_summary) == 1


if __name__ == '__main__':
    test_gleam_sftp_downloader_smoke(); print('GLEAM monthly SFTP smoke test finished.')
