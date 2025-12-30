# -*- coding: utf-8 -*-
"""
premise.nc
==========

NetCDF basic operations for PREMISE v1.0:
- Rename variables / dimensions
- Set attrs / units
- Standardize CF-like structure (time/lat/lon, lon range, lat order)
- Basic subsetting (bbox)
- Concatenate along time
- Write NetCDF with compression

Design goal: minimal user configuration + safe defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr


PathLike = Union[str, Path]


# ---------------------------
# I/O helpers
# ---------------------------

def open_nc(path: PathLike, *, chunks: Optional[Union[str, Dict[str, int]]] = None, decode_times: bool = True) -> xr.Dataset:
    """
    Open a NetCDF file with xarray.

    Parameters
    ----------
    chunks : None | "auto" | dict
        If provided and dask is available, enable lazy loading.
    """
    return xr.open_dataset(path, chunks=chunks, decode_times=decode_times)


def _default_encoding(ds: xr.Dataset, *, comp_level: int = 4) -> Dict[str, Dict[str, Any]]:
    enc: Dict[str, Dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        if da.dtype.kind in "fiu":
            enc[name] = {"zlib": True, "complevel": int(comp_level)}
            # keep existing _FillValue if exists in encoding / attrs
            fv = da.encoding.get("_FillValue", da.attrs.get("_FillValue", None))
            if fv is not None:
                enc[name]["_FillValue"] = fv
    return enc


def to_netcdf(
    ds_or_path: Union[xr.Dataset, PathLike],
    out_nc: PathLike,
    *,
    comp_level: int = 4,
    engine: str = "netcdf4",
    format: str = "NETCDF4",
    encoding: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """
    Write Dataset to NetCDF with safe compression defaults.
    """
    if isinstance(ds_or_path, (str, Path)):
        ds = open_nc(ds_or_path)
    else:
        ds = ds_or_path

    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    enc = encoding if encoding is not None else _default_encoding(ds, comp_level=comp_level)
    ds.to_netcdf(out_nc, engine=engine, format=format, encoding=enc)
    return str(out_nc)


# ---------------------------
# Basic ops
# ---------------------------

def rename_vars(
    ds_or_path: Union[xr.Dataset, PathLike],
    mapping: Mapping[str, str],
    *,
    out_nc: Optional[PathLike] = None,
    strict: bool = False,
    comp_level: int = 4,
) -> xr.Dataset:
    """
    Rename variables in a Dataset (or a file).

    Parameters
    ----------
    strict : if True, raise when a key is not found
    """
    ds = open_nc(ds_or_path) if isinstance(ds_or_path, (str, Path)) else ds_or_path

    missing = [k for k in mapping.keys() if k not in ds.variables]
    if missing and strict:
        raise KeyError(f"Variables not found: {missing}")

    rename_map = {k: v for k, v in mapping.items() if k in ds.variables}
    ds2 = ds.rename(rename_map)

    if out_nc is not None:
        to_netcdf(ds2, out_nc, comp_level=comp_level)
    return ds2


def rename_dims(
    ds_or_path: Union[xr.Dataset, PathLike],
    mapping: Mapping[str, str],
    *,
    out_nc: Optional[PathLike] = None,
    strict: bool = False,
    comp_level: int = 4,
) -> xr.Dataset:
    """
    Rename dimensions (and corresponding coordinates if present).
    """
    ds = open_nc(ds_or_path) if isinstance(ds_or_path, (str, Path)) else ds_or_path

    missing = [k for k in mapping.keys() if k not in ds.dims]
    if missing and strict:
        raise KeyError(f"Dimensions not found: {missing}")

    rename_map = {k: v for k, v in mapping.items() if k in ds.dims}
    ds2 = ds.rename_dims(rename_map)

    # If coord names equal dim names, rename coords too
    coord_rename = {k: v for k, v in rename_map.items() if k in ds2.coords}
    if coord_rename:
        ds2 = ds2.rename(coord_rename)

    if out_nc is not None:
        to_netcdf(ds2, out_nc, comp_level=comp_level)
    return ds2


def set_var_attrs(
    ds_or_path: Union[xr.Dataset, PathLike],
    var: str,
    attrs: Mapping[str, Any],
    *,
    out_nc: Optional[PathLike] = None,
    comp_level: int = 4,
) -> xr.Dataset:
    """
    Update a variable's attributes.
    """
    ds = open_nc(ds_or_path) if isinstance(ds_or_path, (str, Path)) else ds_or_path
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset.")

    ds2 = ds.copy()
    ds2[var].attrs.update(dict(attrs))

    if out_nc is not None:
        to_netcdf(ds2, out_nc, comp_level=comp_level)
    return ds2


def set_global_attrs(
    ds_or_path: Union[xr.Dataset, PathLike],
    attrs: Mapping[str, Any],
    *,
    out_nc: Optional[PathLike] = None,
    comp_level: int = 4,
) -> xr.Dataset:
    ds = open_nc(ds_or_path) if isinstance(ds_or_path, (str, Path)) else ds_or_path
    ds2 = ds.copy()
    ds2.attrs.update(dict(attrs))
    if out_nc is not None:
        to_netcdf(ds2, out_nc, comp_level=comp_level)
    return ds2


def keep_vars(
    ds_or_path: Union[xr.Dataset, PathLike],
    variables: Sequence[str],
    *,
    out_nc: Optional[PathLike] = None,
    comp_level: int = 4,
) -> xr.Dataset:
    ds = open_nc(ds_or_path) if isinstance(ds_or_path, (str, Path)) else ds_or_path
    missing = [v for v in variables if v not in ds]
    if missing:
        raise KeyError(f"Variables not found: {missing}")
    ds2 = ds[variables]
    if out_nc is not None:
        to_netcdf(ds2, out_nc, comp_level=comp_level)
    return ds2


def drop_vars(
    ds_or_path: Union[xr.Dataset, PathLike],
    variables: Sequence[str],
    *,
    out_nc: Optional[PathLike] = None,
    comp_level: int = 4,
) -> xr.Dataset:
    ds = open_nc(ds_or_path) if isinstance(ds_or_path, (str, Path)) else ds_or_path
    ds2 = ds.drop_vars([v for v in variables if v in ds.variables], errors="ignore")
    if out_nc is not None:
        to_netcdf(ds2, out_nc, comp_level=comp_level)
    return ds2


# ---------------------------
# CF-like standardization
# ---------------------------

_LAT_CANDIDATES = ("lat", "latitude", "y", "nav_lat")
_LON_CANDIDATES = ("lon", "longitude", "x", "nav_lon")
_TIME_CANDIDATES = ("time", "Times", "date", "datetime")


def _guess_coord_name(ds: xr.Dataset, kind: str) -> Optional[str]:
    if kind == "lat":
        cands = _LAT_CANDIDATES
    elif kind == "lon":
        cands = _LON_CANDIDATES
    elif kind == "time":
        cands = _TIME_CANDIDATES
    else:
        raise ValueError("kind must be one of: lat, lon, time")

    # prioritize coords, then dims
    for n in cands:
        if n in ds.coords:
            return n
    for n in cands:
        if n in ds.dims:
            return n
    return None


def _guess_main_var(ds: xr.Dataset) -> str:
    """
    Heuristic: choose a data_var with (time?, lat, lon) dims preference.
    """
    lat = _guess_coord_name(ds, "lat")
    lon = _guess_coord_name(ds, "lon")
    time = _guess_coord_name(ds, "time")

    best = None
    best_score = -1
    for v in ds.data_vars:
        dims = ds[v].dims
        score = 0
        if lat and lat in dims:
            score += 2
        if lon and lon in dims:
            score += 2
        if time and time in dims:
            score += 1
        # prefer 2D/3D grids
        if len(dims) in (2, 3):
            score += 1
        if score > best_score:
            best_score = score
            best = v

    if best is None:
        raise ValueError("Cannot guess main variable. Please specify var explicitly.")
    return best


def _fix_lon_range(lon: xr.DataArray, target: str) -> xr.DataArray:
    """
    target: "0_360" or "-180_180"
    """
    lon_vals = lon.values
    if not np.issubdtype(lon_vals.dtype, np.number):
        return lon

    if target == "0_360":
        lon2 = (lon_vals + 360.0) % 360.0
        return lon.copy(data=lon2)
    if target == "-180_180":
        lon2 = ((lon_vals + 180.0) % 360.0) - 180.0
        return lon.copy(data=lon2)
    raise ValueError("target must be '0_360' or '-180_180'")


def _convert_pr_units(values: xr.DataArray, from_units: str, to_units: str) -> xr.DataArray:
    """
    Minimal precipitation unit conversion.
    Supported:
      - kg m-2 s-1  <->  mm day-1 (assuming density 1000 kg/m3 so 1 kg/m2 = 1 mm water)
      - mm/s -> mm/day
      - mm/hr -> mm/day
      - mm/day -> mm/day (no-op)
    """
    fu = (from_units or "").strip().lower().replace(" ", "")
    tu = (to_units or "").strip().lower().replace(" ", "")

    if fu == tu:
        return values

    # normalize common spellings
    fu = fu.replace("mmday-1", "mm/day").replace("mm/d", "mm/day")
    tu = tu.replace("mmday-1", "mm/day").replace("mm/d", "mm/day")

    if fu in ("kgm-2s-1", "kg/m^2/s", "kgm^-2s^-1", "kgm-2s-1") and tu == "mm/day":
        return values * 86400.0
    if fu in ("mm/s", "mms-1") and tu == "mm/day":
        return values * 86400.0
    if fu in ("mm/hr", "mm/h") and tu == "mm/day":
        return values * 24.0
    if fu == "mm/day" and tu == "kgm-2s-1":
        return values / 86400.0

    # if not supported, return unchanged but keep attrs handled by caller
    return values


@dataclass
class StandardizeCFOptions:
    target_time: str = "time"
    target_lat: str = "lat"
    target_lon: str = "lon"
    lon_range: str = "-180_180"   # "-180_180" or "0_360"
    sort_lat_ascending: bool = True
    sort_lon_ascending: bool = True
    target_var: Optional[str] = None   # final variable name, e.g. "pr"
    source_var: Optional[str] = None   # if known
    target_units: Optional[str] = None # e.g. "mm day-1" or "mm/day"
    set_cf_coord_attrs: bool = True


def standardize_cf(
    ds_or_path: Union[xr.Dataset, PathLike],
    *,
    options: Optional[StandardizeCFOptions] = None,
    out_nc: Optional[PathLike] = None,
    comp_level: int = 4,
) -> xr.Dataset:
    """
    Standardize a dataset towards CF-like conventions:
      - rename coord/dims to time/lat/lon if possible
      - fix lon range to -180..180 or 0..360
      - sort lat/lon
      - optionally rename main variable and convert its units (precip-only minimal conversion)
    """
    ds = open_nc(ds_or_path) if isinstance(ds_or_path, (str, Path)) else ds_or_path
    opt = options or StandardizeCFOptions()

    ds2 = ds.copy()

    # rename coords/dims
    lat0 = _guess_coord_name(ds2, "lat")
    lon0 = _guess_coord_name(ds2, "lon")
    time0 = _guess_coord_name(ds2, "time")

    rename_vars_map: Dict[str, str] = {}
    rename_dims_map: Dict[str, str] = {}

    if lat0 and lat0 != opt.target_lat:
        if lat0 in ds2.dims:
            rename_dims_map[lat0] = opt.target_lat
        if lat0 in ds2.coords:
            rename_vars_map[lat0] = opt.target_lat

    if lon0 and lon0 != opt.target_lon:
        if lon0 in ds2.dims:
            rename_dims_map[lon0] = opt.target_lon
        if lon0 in ds2.coords:
            rename_vars_map[lon0] = opt.target_lon

    if time0 and time0 != opt.target_time:
        if time0 in ds2.dims:
            rename_dims_map[time0] = opt.target_time
        if time0 in ds2.coords:
            rename_vars_map[time0] = opt.target_time

    if rename_dims_map:
        ds2 = ds2.rename_dims(rename_dims_map)
    if rename_vars_map:
        ds2 = ds2.rename(rename_vars_map)

    # ensure coords are present
    if opt.target_lat not in ds2.coords and opt.target_lat in ds2.dims:
        ds2 = ds2.assign_coords({opt.target_lat: np.arange(ds2.dims[opt.target_lat], dtype=float)})
    if opt.target_lon not in ds2.coords and opt.target_lon in ds2.dims:
        ds2 = ds2.assign_coords({opt.target_lon: np.arange(ds2.dims[opt.target_lon], dtype=float)})

    # lon range fix + sort lon
    if opt.target_lon in ds2.coords:
        lon = ds2[opt.target_lon]
        lon2 = _fix_lon_range(lon, opt.lon_range)
        ds2 = ds2.assign_coords({opt.target_lon: lon2})
        if opt.sort_lon_ascending:
            ds2 = ds2.sortby(opt.target_lon)

    # sort lat
    if opt.target_lat in ds2.coords and opt.sort_lat_ascending:
        ds2 = ds2.sortby(opt.target_lat)

    # set CF coord attrs
    if opt.set_cf_coord_attrs:
        if opt.target_lat in ds2.coords:
            ds2[opt.target_lat].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
        if opt.target_lon in ds2.coords:
            ds2[opt.target_lon].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
        if opt.target_time in ds2.coords:
            ds2[opt.target_time].attrs.setdefault("standard_name", "time")

    # variable rename + unit conversion (optional)
    src_var = opt.source_var or (opt.target_var if opt.target_var in ds2.data_vars else None)
    if src_var is None:
        src_var = _guess_main_var(ds2)

    if opt.target_units is not None:
        fu = ds2[src_var].attrs.get("units", "")
        da_new = _convert_pr_units(ds2[src_var], fu, opt.target_units)
        ds2[src_var] = da_new
        ds2[src_var].attrs["units"] = opt.target_units

    if opt.target_var is not None and src_var != opt.target_var:
        ds2 = ds2.rename({src_var: opt.target_var})

    if out_nc is not None:
        to_netcdf(ds2, out_nc, comp_level=comp_level)
    return ds2


# ---------------------------
# Subsetting and concat
# ---------------------------

def subset_bbox(
    ds_or_path: Union[xr.Dataset, PathLike],
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    out_nc: Optional[PathLike] = None,
    comp_level: int = 4,
) -> xr.Dataset:
    ds = open_nc(ds_or_path) if isinstance(ds_or_path, (str, Path)) else ds_or_path

    lon_name = _guess_coord_name(ds, "lon") or "lon"
    lat_name = _guess_coord_name(ds, "lat") or "lat"

    ds2 = ds.sel(
        **{
            lon_name: slice(lon_min, lon_max),
            lat_name: slice(lat_min, lat_max),
        }
    )
    if out_nc is not None:
        to_netcdf(ds2, out_nc, comp_level=comp_level)
    return ds2


def concat_time(
    paths: Sequence[PathLike],
    *,
    out_nc: Optional[PathLike] = None,
    comp_level: int = 4,
    chunks: Optional[Union[str, Dict[str, int]]] = None,
) -> xr.Dataset:
    """
    Concatenate multiple NetCDF files along time.

    Notes:
    - assumes shared spatial grid
    - uses xarray.open_mfdataset when possible
    """
    pths = [str(p) for p in paths]
    ds = xr.open_mfdataset(pths, combine="by_coords", chunks=chunks)
    if out_nc is not None:
        to_netcdf(ds, out_nc, comp_level=comp_level)
    return ds
