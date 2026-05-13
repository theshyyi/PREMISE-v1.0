from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr

from ..core.netcdf import to_netcdf as _save_netcdf


def _read_hdf_dataset(handle: h5py.File, path: str | None) -> np.ndarray:
    if path is None:
        candidates: list[str] = []

        def _visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                candidates.append(name)

        handle.visititems(_visit)
        candidates = [
            c for c in candidates
            if not c.lower().endswith(("lat", "latitude", "lon", "longitude", "time"))
        ]
        if len(candidates) != 1:
            raise ValueError(
                "HDF dataset path is ambiguous. Please provide `dataset=...`. "
                f"Available data candidates: {candidates}"
            )
        path = candidates[0]
    return np.asarray(handle[path])


def _maybe_fix_2d_orientation(
    arr: np.ndarray,
    lat: np.ndarray | None,
    lon: np.ndarray | None,
) -> np.ndarray:
    """
    自动判断二维 HDF 数据是 (lat, lon) 还是 (lon, lat)。
    如果发现当前 shape 与 lat/lon 长度反过来，就自动转置。
    """
    if arr.ndim != 2:
        return arr

    n0, n1 = arr.shape
    lat_len = None if lat is None else len(lat)
    lon_len = None if lon is None else len(lon)

    # 标准情况：arr.shape == (lat, lon)
    if lat_len == n0 and lon_len == n1:
        return arr

    # IMERG 等常见情况：arr.shape == (lon, lat)
    if lat_len == n1 and lon_len == n0:
        return arr.T

    return arr


def _maybe_fix_3d_orientation(
    arr: np.ndarray,
    lat: np.ndarray | None,
    lon: np.ndarray | None,
) -> np.ndarray:
    """
    自动判断三维数据是 (time, lat, lon) 还是 (time, lon, lat)。
    若后两维与 lat/lon 长度相反，则交换后两维。
    """
    if arr.ndim != 3:
        return arr

    nt, n1, n2 = arr.shape
    lat_len = None if lat is None else len(lat)
    lon_len = None if lon is None else len(lon)

    # 标准情况：(time, lat, lon)
    if lat_len == n1 and lon_len == n2:
        return arr

    # 常见反置情况：(time, lon, lat)
    if lat_len == n2 and lon_len == n1:
        return np.transpose(arr, (0, 2, 1))

    return arr


def open_hdf(
    path: str | Path,
    *,
    group: str | None = None,
    dataset: str | None = None,
    var_name: str = "var",
    units: str | None = None,
    nodata: float | None = None,
    lat_path: str | None = None,
    lon_path: str | None = None,
    time_path: str | None = None,
) -> xr.Dataset:
    path = Path(path)

    with h5py.File(path, "r") as f:
        root = f[group] if group else f
        arr = _read_hdf_dataset(root, dataset)

        if nodata is None and dataset is not None:
            try:
                nodata = root[dataset].attrs.get("_FillValue")
            except Exception:
                nodata = None

        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        lat = np.asarray(root[lat_path]) if lat_path else None
        lon = np.asarray(root[lon_path]) if lon_path else None
        time = np.asarray(root[time_path]) if time_path else None

        # 自动修正轴顺序
        if arr.ndim == 2:
            arr = _maybe_fix_2d_orientation(arr, lat, lon)
            dims = ("lat", "lon")

        elif arr.ndim == 3:
            arr = _maybe_fix_3d_orientation(arr, lat, lon)
            dims = ("time", "lat", "lon")

        else:
            raise ValueError(f"Only 2D/3D HDF datasets are supported, got shape={arr.shape!r}")

        coords = {}

        if "time" in dims:
            if time is not None:
                coords["time"] = time
            else:
                coords["time"] = np.arange(arr.shape[0])

        if "lat" in dims:
            if lat is not None:
                coords["lat"] = lat
            else:
                coords["lat"] = np.arange(arr.shape[-2])

        if "lon" in dims:
            if lon is not None:
                coords["lon"] = lon
            else:
                coords["lon"] = np.arange(arr.shape[-1])

    da = xr.DataArray(arr, dims=dims, coords=coords, name=var_name)

    if units is not None:
        da.attrs["units"] = units

    return da.to_dataset(name=var_name)


def hdf_to_netcdf(
    path: str | Path,
    out_nc: str | Path,
    *,
    group: str | None = None,
    dataset: str | None = None,
    var_name: str = "var",
    units: str | None = None,
    nodata: float | None = None,
    lat_path: str | None = None,
    lon_path: str | None = None,
    time_path: str | None = None,
    comp_level: int = 4,
) -> str:
    ds = open_hdf(
        path,
        group=group,
        dataset=dataset,
        var_name=var_name,
        units=units,
        nodata=nodata,
        lat_path=lat_path,
        lon_path=lon_path,
        time_path=time_path,
    )
    encoding = {var_name: {"zlib": True, "complevel": int(comp_level)}} if ds[var_name].dtype.kind in "fiu" else {}
    return _save_netcdf(ds, out_nc, encoding=encoding)