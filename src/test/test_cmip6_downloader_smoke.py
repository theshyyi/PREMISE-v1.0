import json
from pathlib import Path

import pandas as pd

from premise.acquisition import DownloadRequest, download_dataset, list_sources


def test_cmip6_esgf_smoke():
    candidates = [s for s in list_sources(variable="pr") if s.key == "cmip6_esgf"]
    assert len(candidates) == 1, "CMIP6 ESGF source was not found in catalog."

    outdir = Path(r"G:\论文手稿（ESR）\论文EMS\PREMISE-v1.0\tests_output\cmip6_esgf_smoke")

    req = DownloadRequest(
        source_key="cmip6_esgf",
        variables=("pr",),
        start_date="2015-01-01",
        end_date="2015-12-31",
        target_dir=str(outdir),
        notes="smoke test for CMIP6 ESGF search downloader",
        frequency="mon",
        format_preference="NetCDF",
    )

    files = download_dataset(
        req,
        timeout=120,
        max_retries=2,
        limit=20,
        search_type="File",
        facets={
            "model": "ACCESS-ESM1-5",
            "experiment": "historical",
            "member": "r1i1p1f1",
            "table": "Amon",
        },
    )

    assert isinstance(files, list)
    assert len(files) == 3

    json_path = outdir / "search_results.json"
    csv_path = outdir / "search_results.csv"
    wget_path = outdir / "wget_script_url.txt"
    manifest_path = outdir / "acquisition_manifest.json"
    summary_path = outdir / "acquisition_summary.csv"

    assert json_path.exists()
    assert csv_path.exists()
    assert wget_path.exists()
    assert manifest_path.exists()
    assert summary_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["source_key"] == "cmip6_esgf"
    assert manifest["provider_name"] == "ESGF"
    assert manifest["requested_file_count"] >= 1

    df = pd.read_csv(csv_path)
    assert "variable" in df.columns
    assert "model" in df.columns
    assert "experiment" in df.columns

    wget_url = wget_path.read_text(encoding="utf-8").strip()
    assert wget_url.startswith("https://") or wget_url.startswith("http://")


if __name__ == "__main__":
    test_cmip6_esgf_smoke()
    print("CMIP6 ESGF smoke test finished.")