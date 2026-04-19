from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import xarray as xr

from ..core.netcdf import to_netcdf as _save_netcdf


def open_grib(
    path: str | Path,
    *,
    var: Optional[str] = None,
    filter_by_keys: Optional[Dict[str, Any]] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
) -> xr.Dataset:
    kwargs: Dict[str, Any] = dict(backend_kwargs or {})
    if filter_by_keys is not None:
        kwargs["filter_by_keys"] = filter_by_keys
    try:
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=kwargs or None)
    except Exception as e:
        raise ImportError(
            "GRIB support requires cfgrib and eccodes. "
            "Install them or monkeypatch xarray.open_dataset in tests."
        ) from e
    return ds if (var is None) else ds[[var]]


def grib_to_netcdf(
    path: str | Path,
    out_nc: str | Path,
    *,
    var: Optional[str] = None,
    filter_by_keys: Optional[Dict[str, Any]] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    comp_level: int = 4,
) -> str:
    ds = open_grib(path, var=var, filter_by_keys=filter_by_keys, backend_kwargs=backend_kwargs)
    encoding = {}
    for name in ds.data_vars:
        if ds[name].dtype.kind in "fiu":
            encoding[name] = {"zlib": True, "complevel": int(comp_level)}
    return _save_netcdf(ds, out_nc, encoding=encoding)
