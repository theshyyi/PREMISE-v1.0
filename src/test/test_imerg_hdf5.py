import json
import os
from pathlib import Path

import pandas as pd

from premise.acquisition import DownloadRequest, download_dataset, list_sources


def test_imerg_hdf5_downloader_smoke():
    user = os.environ.get('IMERG_PPS_USERNAME')
    passwd = os.environ.get('IMERG_PPS_PASSWORD')
    user = "lexinlong9@gmail.com"
    passwd = "lexinlong9@gmail.com"
    if not user or not passwd:
        print('SKIPPED: set IMERG_PPS_USERNAME and IMERG_PPS_PASSWORD to run this test.')
        return
    candidates = [s for s in list_sources(variable='precipitation') if s.key == 'imerg_final_hdf5']
    assert len(candidates) == 1
    outdir = Path(r"I:\PREMISE-v1.0\tests_output\imerg_hdf5_smoke")
    req = DownloadRequest(source_key='imerg_final_hdf5', variables=('precipitation',), start_date='2020-02-01', end_date='2020-02-01', target_dir=str(outdir), notes='smoke test for IMERG Final HDF5 downloader', frequency='half-hourly', format_preference='HDF5')
    files = download_dataset(req, timeout=120, max_retries=1, file_limit=1, username=user, password=passwd)
    assert isinstance(files, list) and len(files) >= 1 and all(Path(f).exists() for f in files)
    with open(outdir / 'acquisition_manifest.json', 'r', encoding='utf-8') as f: manifest = json.load(f)
    assert manifest['source_key'] == 'imerg_final_hdf5'
    df_summary = pd.read_csv(outdir / 'acquisition_summary.csv'); assert len(df_summary) == 1


if __name__ == '__main__':
    test_imerg_hdf5_downloader_smoke(); print('IMERG HDF5 smoke test finished.')
