from premise.acquisition.catalog import get_source, list_sources, search_sources
from premise.acquisition.downloaders.http_direct import HTTPDirectDownloader


def test_gpcc_monthly_decade_source_is_resolvable():
    src = get_source("gpcc_full_monthly_v2022_025_nc_gz")
    assert src.method == "http_direct"
    assert src.params["time_expansion"] == "decade"


def test_cmorph_ftp_sources_exist():
    keys = {src.key for src in list_sources(variable="precipitation")}
    assert "cmorph_daily_025deg_ftp" in keys
    assert "cmorph_3hourly_025deg_ftp" in keys
    assert "cmorph_30min_025deg_ftp" in keys


def test_http_direct_decade_expansion_for_gpcc_monthly():
    contexts = HTTPDirectDownloader._expand_contexts(
        start_date="1999-01-01",
        end_date="2001-12-31",
        time_expansion="decade",
        params={"decade_anchor_year": 1891, "decade_length": 10},
    )
    assert contexts == [
        {"start_year": "1991", "end_year": "2000", "period_start": "1991", "period_end": "2000"},
        {"start_year": "2001", "end_year": "2010", "period_start": "2001", "period_end": "2010"},
    ]


def test_search_sources_cpc_and_gpcp():
    cpc_keys = {src.key for src in search_sources("cpc")}
    assert "cpc_global_unified_daily_nc" in cpc_keys
    gpcp_keys = {src.key for src in search_sources("gpcp")}
    assert "gpcp_monthly_v23_ncei" in gpcp_keys


def test_public_monthly_static_sources_are_resolvable():
    for key in (
        "udel_precip_monthly_v501_nc",
        "udel_air_monthly_v501_nc",
        "precl_monthly_05deg_nc",
        "precl_monthly_1deg_nc",
        "precl_monthly_25deg_nc",
    ):
        src = get_source(key)
        assert src.method == "http_direct"
        assert src.params["time_expansion"] == "none"


def test_gpcp_daily_source_uses_http_listing_date_expansion():
    src = get_source("gpcp_daily_cdr_v1_3_nc")
    assert src.method == "http_listing"
    assert src.params["time_expansion"] == "date"
    assert src.params["directory_pattern"] == "{year}/"


def test_era5_single_levels_source_uses_cds_api():
    src = get_source("era5_single_levels_hourly")
    assert src.method == "cds_api"
    assert src.params["dataset_name"] == "reanalysis-era5-single-levels"
