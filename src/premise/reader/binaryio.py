# -*- coding: utf-8 -*-
"""
binaryio
========
Universal binary raster data -> NetCDF conversion tool.

Enhancements for PREMISE v1.0:
- meta can be a meta file path (key=value) OR a dict-like mapping (for registry-based usage)
- if nt==1 and time_start is not provided, try to guess date from filename
"""

from __future__ import annotations

import gzip
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

MetaLike = Union[str, Path, Mapping[str, Any]]


def parse_meta_file(meta_path: str) -> Dict[str, str]:
    """
    Parse simple key=value text configuration files.

    - Ignore empty lines and lines starting with ‘#’ or ‘;’
    - Do not perform type conversion; return all values as strings
    """
    meta: Dict[str, str] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith(";"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def _load_meta(meta_like: MetaLike) -> Dict[str, str]:
    """meta 可以是文件路径（key=value）或 dict；统一转成 Dict[str,str]。"""
    if isinstance(meta_like, (str, Path)):
        return parse_meta_file(str(meta_like))
    return {str(k): str(v) for k, v in dict(meta_like).items()}


def _get_required(meta: Dict[str, str], key: str) -> str:
    if key not in meta:
        raise KeyError(f"Required key '{key}' not found in meta configuration.")
    return meta[key]


def _get_optional(meta: Dict[str, str], key: str, default: Optional[str] = None) -> Optional[str]:
    return meta.get(key, default)


def _build_dtype(meta: Dict[str, str]) -> np.dtype:
    dtype_str = _get_optional(meta, "dtype", "float32").lower()
    endian = _get_optional(meta, "endian", "little").lower()

    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "int16": np.int16,
        "int32": np.int32,
        "uint16": np.uint16,
        "uint8": np.uint8,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Supported: {list(dtype_map.keys())}")

    base = np.dtype(dtype_map[dtype_str])
    if endian == "little":
        return base.newbyteorder("<")
    if endian == "big":
        return base.newbyteorder(">")
    raise ValueError("endian must be 'little' or 'big'.")


def _build_coords(meta: Dict[str, str], nx: int, ny: int, nt: int):
    # lon
    lon_start = _get_optional(meta, "lon_start", None)
    lon_step = _get_optional(meta, "lon_step", None)
    lon_min = _get_optional(meta, "lon_min", None)
    lon_max = _get_optional(meta, "lon_max", None)

    if lon_start is not None and lon_step is not None:
        lon = float(lon_start) + float(lon_step) * np.arange(nx)
    elif lon_min is not None and lon_max is not None:
        lon = np.linspace(float(lon_min), float(lon_max), nx)
    else:
        lon = np.arange(nx, dtype=float)

    # lat
    lat_start = _get_optional(meta, "lat_start", None)
    lat_step = _get_optional(meta, "lat_step", None)
    lat_min = _get_optional(meta, "lat_min", None)
    lat_max = _get_optional(meta, "lat_max", None)

    if lat_start is not None and lat_step is not None:
        lat = float(lat_start) + float(lat_step) * np.arange(ny)
    elif lat_min is not None and lat_max is not None:
        lat = np.linspace(float(lat_min), float(lat_max), ny)
    else:
        lat = np.arange(ny, dtype=float)

    # time
    time_start = _get_optional(meta, "time_start", None)
    time_step_days = float(_get_optional(meta, "time_step_days", "1")) if nt > 1 else 1.0

    if nt == 1 and time_start is None:
        # dummy timestamp for alignment purposes
        time = np.array([np.datetime64("1900-01-01")], dtype="datetime64[ns]")
    else:
        if time_start is None:
            raise ValueError("time_start must be provided when nt > 1")
        t0 = pd.to_datetime(time_start)
        time = pd.date_range(t0, periods=nt, freq=pd.Timedelta(days=time_step_days)).values.astype(
            "datetime64[ns]"
        )

    return lon, lat, time


def _guess_time_from_filename(path: str) -> Optional[np.datetime64]:
    """尝试从文件名解析 YYYYMMDD / YYYY.MM.DD / YYYY-MM-DD 等日期。"""
    name = Path(path).name
    patterns = [
        r"(?P<Y>\d{4})[._-](?P<M>\d{2})[._-](?P<D>\d{2})",
        r"(?P<Y>\d{4})(?P<M>\d{2})(?P<D>\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            y = int(m.group("Y"))
            mo = int(m.group("M"))
            d = int(m.group("D"))
            return np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}")
    return None


def load_binary_to_dataarray(
    meta_path: MetaLike,
    data_path: Optional[str] = None,
) -> xr.DataArray:
    """
    Read a binary file and convert it into a (time, lat, lon) DataArray according to meta description.

    meta_path can be:
      - a meta file path (key=value), or
      - a dict-like mapping (recommended for registry usage)
    """
    meta = _load_meta(meta_path)

    # Determine Data File Path
    if data_path is None:
        data_path = _get_optional(meta, "data_path", None)
        if data_path is None:
            raise ValueError("Either 'data_path' in meta or function argument data_path must be provided.")
    data_path = str(data_path)

    # dimensional information
    nx = int(_get_required(meta, "nx"))
    ny = int(_get_required(meta, "ny"))
    nt = int(_get_optional(meta, "nt", "1"))

    order = _get_optional(meta, "order", "tyx").lower()
    if set(order) != {"t", "y", "x"} or len(order) != 3:
        raise ValueError("order must be a permutation of 't', 'y', 'x', e.g., 'tyx', 'ytx', 'xyt'.")

    header_bytes = int(_get_optional(meta, "header_bytes", "0"))

    dtype = _build_dtype(meta)
    expected_size = nx * ny * nt

    # read binary file
    path_lower = data_path.lower()
    if path_lower.endswith(".gz"):
        with gzip.open(data_path, "rb") as f:
            if header_bytes > 0:
                f.read(header_bytes)
            raw = f.read(expected_size * dtype.itemsize)
            data = np.frombuffer(raw, dtype=dtype)
    else:
        with open(data_path, "rb") as f:
            if header_bytes > 0:
                f.seek(header_bytes, os.SEEK_SET)
            data = np.fromfile(f, dtype=dtype, count=expected_size)

    if data.size != expected_size:
        raise ValueError(
            f"Data size mismatch: expected {expected_size} elements but got {data.size}. "
            f"Check nx/ny/nt/dtype/header_bytes."
        )

    # reshape to (t, y, x)
    shape_map = {"t": nt, "y": ny, "x": nx}
    data = data.reshape(tuple(shape_map[c] for c in order))

    # transpose to (t, y, x)
    axis_map = {c: i for i, c in enumerate(order)}
    data = np.transpose(data, axes=(axis_map["t"], axis_map["y"], axis_map["x"]))

    data = data.astype("float32", copy=False)

    # missing and scaling
    missing_value = _get_optional(meta, "missing_value", None)
    if missing_value is not None:
        mv = float(missing_value)
        data = np.where(data == mv, np.nan, data)

    scale_factor = float(_get_optional(meta, "scale_factor", "1.0"))
    add_offset = float(_get_optional(meta, "add_offset", "0.0"))
    if scale_factor != 1.0 or add_offset != 0.0:
        data = data * scale_factor + add_offset

    # coords
    lon, lat, time = _build_coords(meta, nx=nx, ny=ny, nt=nt)

    # If single-time and time_start not set: try guess date from filename
    if nt == 1 and _get_optional(meta, "time_start", None) is None:
        t_guess = _guess_time_from_filename(data_path)
        if t_guess is not None:
            time = np.array([t_guess], dtype="datetime64[ns]")

    var_name = _get_optional(meta, "var_name", "var")

    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name=var_name,
    )

    # attributes
    da.attrs["long_name"] = _get_optional(meta, "var_long_name", "")
    da.attrs["units"] = _get_optional(meta, "var_units", "")
    if missing_value is not None:
        da.attrs["_FillValue"] = np.float32(float(missing_value))

    # coord attrs (CF-friendly)
    da["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    da["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})

    return da


def convert_binary_to_netcdf(
    meta_path: MetaLike,
    out_nc: str,
    data_path: Optional[str] = None,
    *,
    comp_level: int = 4,
) -> xr.Dataset:
    """
    Generate NetCDF files based on meta description and binary file.

    Returns an xr.Dataset for further processing by the calling end.
    """
    da = load_binary_to_dataarray(meta_path, data_path=data_path)
    var_name = da.name
    ds = da.to_dataset(name=var_name)

    # global attributes (if present)
    meta = _load_meta(meta_path)
    for key in ["title", "source", "history"]:
        if key in meta:
            ds.attrs[key] = meta[key]

    out_dir = os.path.dirname(out_nc)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    encoding = {
        var_name: {
            "zlib": True,
            "complevel": int(comp_level),
            "dtype": "float32",
            "_FillValue": np.float32(ds[var_name].attrs.get("_FillValue", np.nan))
            if "_FillValue" in ds[var_name].attrs
            else np.float32(np.nan),
        }
    }

    ds.to_netcdf(out_nc, format="NETCDF4", engine="netcdf4", encoding=encoding)
    return ds
