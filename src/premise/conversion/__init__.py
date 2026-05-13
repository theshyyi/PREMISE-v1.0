"""Integrated format-conversion module for PREMISЕ."""

from .api import (
    detect_format,
    open_standard_dataset,
    save_standard_dataset,
    convert_to_netcdf,
    convert_binary_to_nc,
    convert_geotiff_to_monthly_nc,
    convert_grib_to_nc,
    convert_hdf_to_nc,
)
from .harmonization import summarize_harmonization_scope
from .registry import register_product, get_product, list_products

__all__ = [
    "detect_format",
    "open_standard_dataset",
    "save_standard_dataset",
    "convert_to_netcdf",
    "convert_binary_to_nc",
    "convert_geotiff_to_monthly_nc",
    "convert_grib_to_nc",
    "convert_hdf_to_nc",
    "summarize_harmonization_scope",
    "register_product",
    "get_product",
    "list_products",
]
