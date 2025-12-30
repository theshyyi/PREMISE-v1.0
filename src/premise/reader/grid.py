# -*- coding: utf-8 -*-
"""
premise.readers.grib
====================

GRIB/GRIB2 -> xarray.Dataset -> NetCDF.

Requirements (optional):
- cfgrib
- eccodes

Design:
- lazy import to avoid hard dependency
- supports selecting variables and filtering GRIB messages
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import xarray as xr


PathLike = Union[str, Path]


def _require_cfgrib() -> None:
    try:
        import cfgrib  # noqa: F401
    except Exception as e:
        raise ImportError(
            "GRIB support requires 'cfgrib' and 'eccodes'. "
            "Install optional deps, e.g. `pip install premise[grib]` "
            "or `pip install cfgrib eccodes`."
        ) from e


def open_grib(
    path: PathLike,
    *,
    var: Optional[str] = None,
    filter_by_keys: Optional[Dict[str, Any]] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    decode_times: bool = True,
) -> xr.Dataset:
    """
    Open a GRIB/GRIB2 file as xr.Dataset.

    Parameters
    ----------
    var : str | None
        Select a specific variable after opening.
    filter_by_keys : dict | None
        cfgrib filter, e.g. {"typeOfLevel":"surface"} or {"shortName":"tp"}.
        Useful when a GRIB contains multiple messages/levels.
    backend_kwargs : dict | None
        Passed to xarray.open_dataset(engine="cfgrib").
    """
    _require_cfgrib()

    p = Path(path)
    bk = dict(backend_kwargs or {})
    if filter_by_keys is not None:
        bk["filter_by_keys"] = dict(filter_by_keys)

    ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs=bk, decode_times=decode_times)

    if var is not None:
        if var not in ds.data_vars:
            raise KeyError(f"Variable '{var}' not found. Available: {list(ds.data_vars)}")
        ds = ds[[var]]

    return ds


def grib_to_netcdf(
    in_grib: PathLike,
    out_nc: PathLike,
    *,
    var: Optional[str] = None,
    filter_by_keys: Optional[Dict[str, Any]] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    comp_level: int = 4,
) -> str:
    """
    Convert GRIB/GRIB2 to NetCDF.

    Notes:
    - For large files, consider chunking via dask on the xarray side if needed.
    """
    ds = open_grib(
        in_grib,
        var=var,
        filter_by_keys=filter_by_keys,
        backend_kwargs=backend_kwargs,
    )

    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    encoding: Dict[str, Dict[str, Any]] = {}
    for v in ds.data_vars:
        if ds[v].dtype.kind in "fiu":
            encoding[v] = {"zlib": True, "complevel": int(comp_level)}

    ds.to_netcdf(out_nc, engine="netcdf4", format="NETCDF4", encoding=encoding)
    return str(out_nc)
