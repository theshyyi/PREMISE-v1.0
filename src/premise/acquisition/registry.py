from __future__ import annotations

from premise.acquisition.downloaders.http_direct import HTTPDirectDownloader
from premise.acquisition.downloaders.http_listing import HTTPListingDownloader
from premise.acquisition.downloaders.erddap import ERDDAPDownloader
from premise.acquisition.downloaders.cds_api import CDSAPIDownloader
from premise.acquisition.downloaders.sftp import SFTPDownloader
from premise.acquisition.downloaders.ftp import FTPDownloader
from premise.acquisition.downloaders.earthdata_cmr import EarthdataCMRDownloader
from premise.acquisition.downloaders.rclone import RcloneDownloader
from premise.acquisition.downloaders.portal_export import PortalExportDownloader
from premise.acquisition.downloaders.esgf import ESGFSearchDownloader
from premise.acquisition.downloaders.wget import ESGFWgetDownloader


METHOD_REGISTRY = {
    "http_direct": HTTPDirectDownloader,
    "http_listing": HTTPListingDownloader,
    "erddap": ERDDAPDownloader,
    "cds_api": CDSAPIDownloader,
    "sftp": SFTPDownloader,
    "ftp": FTPDownloader,
    "earthdata_cmr": EarthdataCMRDownloader,
    "rclone": RcloneDownloader,
    "portal_export": PortalExportDownloader,
    "esgf": ESGFSearchDownloader,
    "esgf_wget": ESGFWgetDownloader,
}


def get_downloader_class_by_method(method: str):
    key = method.lower().strip()
    if key not in METHOD_REGISTRY:
        raise KeyError(f"No downloader registered for method='{method}'")
    return METHOD_REGISTRY[key]
