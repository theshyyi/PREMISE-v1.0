from premise.acquisition import list_sources, search_sources, DownloadRequest, BaseDownloader, build_download_plan, get_downloader_class_by_method


def test_list_sources_precipitation():
    keys = {src.key for src in list_sources(variable="precipitation")}
    assert "chirps_monthly_nc" in keys
    assert "imerg_final_hdf5" in keys
    assert "mswep_monthly_nc" in keys
    assert "persiann_cdr_erddap" in keys
    assert "cmfd" in keys
    assert "gpcc_full_daily_v2022_10_nc_gz" in keys
    assert "cmorph_daily_025deg_ftp" in keys
    assert "gpcp_monthly_v23_ncei" in keys
    assert "gpcp_daily_cdr_v1_3_nc" in keys
    assert "precl_monthly_05deg_nc" in keys
    assert "udel_precip_monthly_v501_nc" in keys


def test_search_sources_soil_moisture():
    keys = {src.key for src in search_sources("soil moisture")}
    assert "gleam_monthly_nc" in keys


def test_search_sources_gpcc():
    keys = {src.key for src in search_sources("gpcc")}
    assert "gpcc_full_daily_v2022_10_nc_gz" in keys
    assert "gpcc_full_monthly_v2022_025_nc_gz" in keys


def test_build_download_plan_preserves_request_information():
    req = DownloadRequest(
        source_key="chirps_monthly_nc",
        variables=("precipitation",),
        start_date="2020-01-01",
        end_date="2020-12-31",
        bbox=(100.0, 20.0, 110.0, 30.0),
        target_dir="./data/raw/chirps",
        notes="test request",
    )
    plan = build_download_plan([req])
    assert len(plan) == 1
    assert plan[0]["source_key"] == "chirps_monthly_nc"
    assert plan[0]["variables"] == ["precipitation"]
    assert plan[0]["start_date"] == "2020-01-01"
    assert plan[0]["end_date"] == "2020-12-31"
    assert plan[0]["bbox"] == (100.0, 20.0, 110.0, 30.0)
    assert plan[0]["target_dir"] == "./data/raw/chirps"


def test_base_downloader_requires_subclass_implementation():
    req = DownloadRequest(source_key="chirps_monthly_nc", variables=("precipitation",))
    downloader = BaseDownloader()
    try:
        downloader.download(req)
        assert False, "BaseDownloader.download() should raise NotImplementedError"
    except NotImplementedError:
        assert True


def test_ftp_method_is_registered():
    downloader_cls = get_downloader_class_by_method("ftp")
    assert downloader_cls.__name__ == "FTPDownloader"



def test_search_sources_era5_and_precl():
    era5_keys = {src.key for src in search_sources("era5")}
    assert "era5land_hourly" in era5_keys
    assert "era5_single_levels_hourly" in era5_keys

    precl_keys = {src.key for src in search_sources("precl")}
    assert "precl_monthly_05deg_nc" in precl_keys
    assert "precl_monthly_1deg_nc" in precl_keys
    assert "precl_monthly_25deg_nc" in precl_keys


def test_search_sources_udel():
    keys = {src.key for src in search_sources("udel")}
    assert "udel_precip_monthly_v501_nc" in keys
    assert "udel_air_monthly_v501_nc" in keys
