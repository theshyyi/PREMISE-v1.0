# -*- coding: utf-8 -*-

"""
io
==

NetCDF read/write and file management utilities.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import xarray as xr
from .io_api import open_dataset, to_netcdf

from .nc_base import open_nc as open_nc
from .nc_base import to_netcdf as to_netcdf


__all__ = [
    # 你原来 io.py 的 __all__ ...
    "open_dataset",
    "to_netcdf",
]


def open_dataset(
    path: str,
    chunks: str | dict | None = "auto",
    decode_times: bool = True,
) -> xr.Dataset:
    """
    Unified entry to open a NetCDF dataset.

    Parameters
    ----------
    path : str
        NetCDF file path.
    chunks : str or dict or None, default "auto"
        Dask chunking option passed to xarray.open_dataset.
    decode_times : bool, default True
        Whether to decode time coordinates.

    Returns
    -------
    ds : xr.Dataset
    """
    return xr.open_dataset(path, chunks=chunks, decode_times=decode_times)


def save_dataset(
    ds: xr.Dataset,
    path: str,
    encoding: Optional[dict] = None,
) -> None:
    """
    Unified entry to save a NetCDF dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be saved.
    path : str
        Output file path.
    encoding : dict or None
        Encoding options for xarray.to_netcdf.
    """
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    ds.to_netcdf(path, encoding=encoding or {})


def list_nc_files(
    directory: str,
    pattern: str = "*.nc",
) -> List[str]:
    """
    List all NetCDF files in a directory matching a wildcard pattern.

    Parameters
    ----------
    directory : str
        Directory path.
    pattern : str, default "*.nc"
        Glob pattern, e.g., "*.TIMEFIX.daily.CHINA.nc".

    Returns
    -------
    paths : list of str
        Sorted list of file paths.
    """
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    return paths
