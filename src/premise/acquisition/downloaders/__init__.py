from premise.acquisition.downloaders.http_direct import HTTPDirectDownloader
from premise.acquisition.downloaders.http_listing import HTTPListingDownloader
from premise.acquisition.downloaders.erddap import ERDDAPDownloader
from premise.acquisition.downloaders.cds_api import CDSAPIDownloader
from premise.acquisition.downloaders.sftp import SFTPDownloader
from premise.acquisition.downloaders.ftp import FTPDownloader
from premise.acquisition.downloaders.rclone import RcloneDownloader
from premise.acquisition.downloaders.portal_export import PortalExportDownloader

__all__ = [
    "HTTPDirectDownloader",
    "HTTPListingDownloader",
    "ERDDAPDownloader",
    "CDSAPIDownloader",
    "SFTPDownloader",
    "FTPDownloader",
    "RcloneDownloader",
    "PortalExportDownloader",
    "EarthdataCMRDownloader",
]

from premise.acquisition.downloaders.earthdata_cmr import EarthdataCMRDownloader
