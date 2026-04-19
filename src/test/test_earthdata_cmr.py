from premise.acquisition.base import DownloadRequest
from premise.acquisition.catalog import get_source, search_sources
from premise.acquisition.downloaders.earthdata_cmr import EarthdataCMRDownloader
from premise.acquisition.registry import get_downloader_class_by_method


def test_earthdata_method_is_registered():
    cls = get_downloader_class_by_method("earthdata_cmr")
    assert cls is EarthdataCMRDownloader


def test_earthdata_sources_are_resolvable():
    expected = {
        "trmm_3b42_daily_v7": "TRMM_3B42_Daily",
        "trmm_3b42_3hourly_v7": "TRMM_3B42",
        "merra2_tavg1_2d_flx_nx_hourly": "M2T1NXFLX",
        "merra2_tavg1_2d_slv_nx_hourly": "M2T1NXSLV",
        "merra2_tavgm_2d_flx_nx_monthly": "M2TMNXFLX",
        "gldas_noah025_3h_v21": "GLDAS_NOAH025_3H",
        "fldas_noah01_c_gl_m_v001": "FLDAS_NOAH01_C_GL_M",
        "fldas_noah01_cp_gl_m_v001": "FLDAS_NOAH01_CP_GL_M",
    }
    for key, short_name in expected.items():
        src = get_source(key)
        assert src.method == "earthdata_cmr"
        assert src.params["short_name"] == short_name


def test_search_sources_earthdata_hits_gldas_and_merra2():
    gldas_keys = {src.key for src in search_sources("gldas")}
    assert "gldas_noah025_3h_v21" in gldas_keys
    merra_keys = {src.key for src in search_sources("merra2")}
    assert "merra2_tavg1_2d_flx_nx_hourly" in merra_keys


def test_build_cmr_query_params_uses_short_name_version_and_bbox():
    src = get_source("gldas_noah025_3h_v21")
    downloader = EarthdataCMRDownloader(source_config=src)
    req = DownloadRequest(
        source_key=src.key,
        variables=("precipitation",),
        start_date="2020-01-01",
        end_date="2020-01-31",
        bbox=(100.0, 20.0, 110.0, 30.0),
        target_dir="./tmp",
    )
    params = downloader._build_cmr_query_params(request=req, page_num=1, page_size=500)
    assert params["short_name"] == "GLDAS_NOAH025_3H"
    assert params["version"] == "2.1"
    assert params["provider"] == "GES_DISC"
    assert params["bounding_box"] == "100.0,20.0,110.0,30.0"
    assert params["temporal"] == "2020-01-01T00:00:00Z,2020-01-31T23:59:59Z"


def test_extract_download_links_filters_metadata_and_keeps_data_link():
    params = {
        "preferred_link_substrings": ["gesdisc"],
        "filename_regex": r"\.(nc4|nc)$",
    }
    entry = {
        "links": [
            {
                "href": "https://cmr.earthdata.nasa.gov/search/concepts/C123.xml",
                "rel": "http://esipfed.org/ns/fedsearch/1.1/metadata#",
                "type": "application/xml",
            },
            {
                "href": "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_3H.2.1/2020/001/GLDAS_NOAH025_3H.A20200101.0000.021.nc4",
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                "type": "application/x-netcdf",
            },
            {
                "href": "https://hydro1.gesdisc.eosdis.nasa.gov/opendap/GLDAS/file.html",
                "rel": "http://esipfed.org/ns/fedsearch/1.1/service#",
                "type": "text/html",
            },
        ]
    }
    hits = EarthdataCMRDownloader._extract_download_links_from_entry(entry=entry, params=params)
    assert hits == [
        (
            "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_3H.2.1/2020/001/GLDAS_NOAH025_3H.A20200101.0000.021.nc4",
            "GLDAS_NOAH025_3H.A20200101.0000.021.nc4",
        )
    ]
