# -*- coding: utf-8 -*-
"""
premise.readers.hdf
===================

HDF5 / NetCDF4-container reading utilities.

Two modes:
1) CF-like HDF5/NetCDF4 container: use xarray (h5netcdf/netcdf4 engines) with near-zero config
2) Generic HDF5: allow selecting an internal dataset path and heuristically build coords

Optional deps:
- h5netcdf OR netcdf4 (recommended)
- h5py (for listing datasets and generic fallback)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr


PathLike = Union[str, Path]


_LAT_CANDS = ("lat", "latitude", "y", "nav_lat")
_LON_CANDS = ("lon", "longitude", "x", "nav_lon")
_TIME_CANDS = ("time", "Times", "date", "datetime")


def _try_open_xarray(path: PathLike, *, group: Optional[str] = None) -> Optional[xr.Dataset]:
    p = Path(path)
    # Try h5netcdf first (often good for HDF5/netCDF4)
    for engine in ("h5netcdf", "netcdf4"):
        try:
            ds = xr.open_dataset(p, engine=engine, group=group) if group else xr.open_dataset(p, engine=engine)
            return ds
        except Exception:
            continue
    return None


def _require_h5py():
    try:
        import h5py  # noqa: F401
        return h5py
    except Exception as e:
        raise ImportError(
            "Generic HDF5 fallback requires 'h5py'. "
            "Install it via `pip install h5py` or provide CF-like HDF readable by xarray engines."
        ) from e


def list_hdf_datasets(path: PathLike) -> List[Tuple[str, Tuple[int, ...], str]]:
    """
    List all datasets inside an HDF5 file: (dataset_path, shape, dtype).
    Useful for users to find the right dataset path for non-CF HDF.
    """
    h5py = _require_h5py()
    p = Path(path)

    out: List[Tuple[str, Tuple[int, ...], str]] = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append((f"/{name}", tuple(obj.shape), str(obj.dtype)))

    with h5py.File(p, "r") as f:
        f.visititems(_visit)
    return out


def _find_first_numeric_dataset(datasets: List[Tuple[str, Tuple[int, ...], str]]) -> Optional[str]:
    # Pick a 2D/3D numeric dataset as a default guess
    for path, shape, dtype in datasets:
        if len(shape) in (2, 3) and any(t in dtype for t in ("float", "int", "uint")):
            return path
    return datasets[0][0] if datasets else None


def _find_coord_path(datasets: List[Tuple[str, Tuple[int, ...], str]], candidates: Sequence[str]) -> Optional[str]:
    # match by trailing name
    for cand in candidates:
        for p, _, _ in datasets:
            if p.lower().endswith("/" + cand.lower()):
                return p
    return None


def open_hdf(
    path: PathLike,
    *,
    group: Optional[str] = None,
    dataset: Optional[str] = None,
    var_name: str = "var",
    units: Optional[str] = None,
    nodata: Optional[float] = None,
    lat_path: Optional[str] = None,
    lon_path: Optional[str] = None,
    time_path: Optional[str] = None,
) -> xr.Dataset:
    """
    Open HDF5 / NetCDF4-container as xr.Dataset.

    Mode A (near-zero config):
      - if file is CF-like, xarray engines can open it directly.

    Mode B (generic HDF5 fallback):
      - choose an internal dataset path, plus optional coord datasets

    Parameters
    ----------
    group : str | None
        For HDF5/NetCDF4 that uses groups.
    dataset : str | None
        Internal HDF dataset path (e.g., "/Grid/precipitation"). Required for generic HDF.
        If None and xarray open fails, we will try to auto-select a 2D/3D dataset.
    lat_path / lon_path / time_path : str | None
        Internal dataset paths for coords. If None, will try to infer by common names.
    """
    p = Path(path)

    # ---- Mode A: try xarray directly ----
    ds = _try_open_xarray(p, group=group)
    if ds is not None:
        return ds

    # ---- Mode B: generic HDF5 fallback ----
    h5py = _require_h5py()
    dsets = list_hdf_datasets(p)
    if not dsets:
        raise ValueError(f"No datasets found inside HDF file: {p}")

    if dataset is None:
        dataset = _find_first_numeric_dataset(dsets)
        if dataset is None:
            raise ValueError("Cannot auto-select a numeric dataset. Please specify 'dataset' explicitly.")

    # infer coord paths if not provided
    if lat_path is None:
        lat_path = _find_coord_path(dsets, _LAT_CANDS)
    if lon_path is None:
        lon_path = _find_coord_path(dsets, _LON_CANDS)
    if time_path is None:
        time_path = _find_coord_path(dsets, _TIME_CANDS)

    with h5py.File(p, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset path not found: {dataset}. Use list_hdf_datasets() to inspect.")
        arr = np.array(f[dataset])

        # coords
        lat = np.array(f[lat_path]) if (lat_path and lat_path in f) else None
        lon = np.array(f[lon_path]) if (lon_path and lon_path in f) else None
        t = np.array(f[time_path]) if (time_path and time_path in f) else None

    # Build coords/dims heuristically
    # Prefer (time, lat, lon) if 3D
    if arr.ndim == 3:
        dims = ("time", "lat", "lon")
        coords: Dict[str, Any] = {}
        if t is not None:
            # try decode numeric time to datetime64? keep raw if unknown
            coords["time"] = t
        else:
            coords["time"] = np.arange(arr.shape[0])
        coords["lat"] = lat if (lat is not None and lat.ndim == 1 and lat.size == arr.shape[1]) else np.arange(arr.shape[1])
        coords["lon"] = lon if (lon is not None and lon.ndim == 1 and lon.size == arr.shape[2]) else np.arange(arr.shape[2])

    elif arr.ndim == 2:
        dims = ("lat", "lon")
        coords = {
            "lat": lat if (lat is not None and lat.ndim == 1 and lat.size == arr.shape[0]) else np.arange(arr.shape[0]),
            "lon": lon if (lon is not None and lon.ndim == 1 and lon.size == arr.shape[1]) else np.arange(arr.shape[1]),
        }
        # promote to time=1 to align with other pipeline components (optional)
        arr = arr[None, ...]
        dims = ("time", "lat", "lon")
        coords = {"time": np.array([0]), **coords}

    else:
        raise ValueError(f"Unsupported dataset ndim={arr.ndim}. Only 2D/3D arrays are supported in fallback mode.")

    da = xr.DataArray(arr, dims=dims, coords=coords, name=var_name).astype("float32", copy=False)

    if nodata is not None:
        da = da.where(da != float(nodata))

    if units is not None:
        da.attrs["units"] = units

    # CF-like coord attrs
    if "lat" in da.coords:
        da["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    if "lon" in da.coords:
        da["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})

    ds_out = da.to_dataset()

    return ds_out


def hdf_to_netcdf(
    in_hdf: PathLike,
    out_nc: PathLike,
    *,
    group: Optional[str] = None,
    dataset: Optional[str] = None,
    var_name: str = "var",
    units: Optional[str] = None,
    nodata: Optional[float] = None,
    lat_path: Optional[str] = None,
    lon_path: Optional[str] = None,
    time_path: Optional[str] = None,
    comp_level: int = 4,
) -> str:
    ds = open_hdf(
        in_hdf,
        group=group,
        dataset=dataset,
        var_name=var_name,
        units=units,
        nodata=nodata,
        lat_path=lat_path,
        lon_path=lon_path,
        time_path=time_path,
    )

    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    encoding: Dict[str, Dict[str, Any]] = {}
    for v in ds.data_vars:
        if ds[v].dtype.kind in "fiu":
            encoding[v] = {"zlib": True, "complevel": int(comp_level)}

    ds.to_netcdf(out_nc, engine="netcdf4", format="NETCDF4", encoding=encoding)
    return str(out_nc)
