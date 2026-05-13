from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import xarray as xr

from ..core.netcdf import to_netcdf as _save_netcdf


def _load_meta(meta: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(meta, dict):
        return dict(meta)
    path = Path(meta)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_coords(meta: Dict[str, Any], shape: tuple[int, ...]) -> Dict[str, Any]:
    if len(shape) == 3:
        nt, ny, nx = shape
    elif len(shape) == 2:
        nt, ny, nx = None, shape[0], shape[1]
    else:
        raise ValueError(f"Only 2D/3D raw arrays are supported, got shape={shape!r}")

    coords: Dict[str, Any] = {}
    if nt is not None:
        if "time" in meta:
            coords["time"] = meta["time"]
        else:
            coords["time"] = np.arange(nt)

    if "lat" in meta:
        coords["lat"] = np.asarray(meta["lat"])
    elif "lat_start" in meta and "lat_step" in meta:
        coords["lat"] = np.asarray(meta["lat_start"] + np.arange(ny) * meta["lat_step"])
    else:
        coords["lat"] = np.arange(ny)

    if "lon" in meta:
        coords["lon"] = np.asarray(meta["lon"])
    elif "lon_start" in meta and "lon_step" in meta:
        coords["lon"] = np.asarray(meta["lon_start"] + np.arange(nx) * meta["lon_step"])
    else:
        coords["lon"] = np.arange(nx)
    return coords


def load_binary_to_dataarray(meta: str | Path | Dict[str, Any], *, data_path: str | Path | None = None) -> xr.DataArray:
    meta = _load_meta(meta)
    data_path = Path(data_path or meta["data_path"])
    dtype = np.dtype(meta.get("dtype", "float32"))
    shape = tuple(int(v) for v in meta["shape"])
    order = meta.get("order", "C")
    var_name = meta.get("var_name", meta.get("name", "var"))
    units = meta.get("units")
    nodata = meta.get("nodata")

    arr = np.fromfile(data_path, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"Binary size mismatch: got {arr.size} elements, expected {expected}")
    arr = arr.reshape(shape, order=order)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    coords = _build_coords(meta, arr.shape)
    dims = ("time", "lat", "lon") if arr.ndim == 3 else ("lat", "lon")
    da = xr.DataArray(arr, dims=dims, coords={k: coords[k] for k in dims}, name=var_name)
    if units is not None:
        da.attrs["units"] = units
    return da


def binary_to_netcdf(
    data_path: str | Path,
    out_nc: str | Path,
    *,
    meta: str | Path | Dict[str, Any],
    comp_level: int = 4,
) -> str:
    da = load_binary_to_dataarray(meta, data_path=data_path)
    encoding = {
        da.name: {"zlib": True, "complevel": int(comp_level)} if da.dtype.kind in "fiu" else {}
    }
    return _save_netcdf(da.to_dataset(name=da.name), out_nc, encoding=encoding)
