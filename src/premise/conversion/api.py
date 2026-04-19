from __future__ import annotations

from .dispatch import open_standard_dataset, save_standard_dataset, convert_to_netcdf, detect_format
from .formats.binary import binary_to_netcdf as convert_binary_to_nc
from .formats.geotiff import geotiff_to_monthly_netcdf as convert_geotiff_to_monthly_nc
from .formats.grib import grib_to_netcdf as convert_grib_to_nc
from .formats.hdf import hdf_to_netcdf as convert_hdf_to_nc

__all__ = [
    "detect_format",
    "open_standard_dataset",
    "save_standard_dataset",
    "convert_to_netcdf",
    "convert_binary_to_nc",
    "convert_geotiff_to_monthly_nc",
    "convert_grib_to_nc",
    "convert_hdf_to_nc",
]
