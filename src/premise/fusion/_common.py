# -*- coding: utf-8 -*-
"""
premise.fusion._common
======================

Shared utilities for fusion submodules (PREMISE v1.0).

Goals:
- Keep heavy dependencies optional (geopandas/regionmask/sklearn/joblib/tqdm)
- Robust handling of lat/lon/time coordinate variants and cftime objects
- Minimal user configuration via JSON
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


def require_optional(pkgs: List[str], extra_hint: str = "fusion") -> None:
    """
    Runtime check for optional dependencies. Import names must be importable.
    """
    missing = []
    for p in pkgs:
        try:
            __import__(p)
        except Exception:
            missing.append(p)
    if missing:
        raise ImportError(
            "This fusion component requires optional dependencies: "
            + ", ".join(missing)
            + f". Install, for example: `pip install premise[{extra_hint}]`."
        )


def ensure_dir(p: PathLike) -> Path:
    p = Path(p).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: PathLike) -> dict:
    p = Path(path).expanduser().resolve()
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def as_config(config: Union[dict, str, Path]) -> dict:
    if isinstance(config, dict):
        return config
    return load_json(config)


def ensure_latlon(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for k in list(ds.coords) + list(ds.dims):
        lk = k.lower()
        if lk in ("longitude", "long", "x"):
            rename[k] = "lon"
        if lk in ("latitude", "lati", "y"):
            rename[k] = "lat"
    return ds.rename(rename) if rename else ds


def _to_timestamp(t) -> pd.Timestamp:
    # cftime-like
    if hasattr(t, "year") and hasattr(t, "month") and hasattr(t, "day"):
        hh = getattr(t, "hour", 0)
        mm = getattr(t, "minute", 0)
        ss = getattr(t, "second", 0)
        return pd.Timestamp(int(t.year), int(t.month), int(t.day), int(hh), int(mm), int(ss))
    return pd.Timestamp(t)


def normalize_time_to_daily_index(time_values, floor: Optional[str] = "D") -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex([_to_timestamp(t) for t in time_values])
    return idx.floor(floor) if floor else idx


def standardize_time(da: xr.DataArray, floor: str = "D") -> xr.DataArray:
    if "time" not in da.dims:
        return da
    idx = normalize_time_to_daily_index(da["time"].values, floor=floor)
    return da.assign_coords(time=idx)


def intersect_datetimes(*idxs: pd.DatetimeIndex) -> pd.DatetimeIndex:
    inter = idxs[0]
    for x in idxs[1:]:
        inter = inter.intersection(x)
    return inter


def open_da(path: PathLike, var: Optional[str], chunks: Optional[dict] = None) -> xr.DataArray:
    ds = xr.open_dataset(path, chunks=chunks, decode_times=True)
    ds = ensure_latlon(ds)
    if var is None:
        if len(ds.data_vars) != 1:
            raise ValueError(f"{path}: var not given and multiple vars exist: {list(ds.data_vars)}")
        var = list(ds.data_vars)[0]
    if var not in ds.data_vars:
        raise KeyError(f"{path}: variable '{var}' not found. Available: {list(ds.data_vars)}")
    da = ds[var]
    if "time" in da.dims:
        da = da.transpose("time", "lat", "lon")
    else:
        da = da.transpose("lat", "lon")
    return da


def align_to_ref_grid(da: xr.DataArray, ref: xr.DataArray, method: str = "linear") -> xr.DataArray:
    return da.interp(lat=ref["lat"], lon=ref["lon"], method=method)


def build_climate_id_mask(shp_path: PathLike, field: str, ref: xr.DataArray, force_epsg_if_missing: Optional[int] = None):
    """
    Generate climate_id mask from polygons, aligned to ref grid.
    Returns:
      climate_id: xr.DataArray(lat, lon), float32, 1..N, NaN outside
      mapping: dict[int,str]
    """
    require_optional(["geopandas", "regionmask"], extra_hint="fusion")
    import geopandas as gpd
    import regionmask

    gdf = gpd.read_file(str(shp_path))
    if gdf.crs is None and force_epsg_if_missing is not None:
        gdf = gdf.set_crs(epsg=int(force_epsg_if_missing), allow_override=True)

    # try project to WGS84
    try:
        gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        pass

    if field not in gdf.columns:
        raise KeyError(f"Field '{field}' not found in shapefile. Available: {list(gdf.columns)}")

    gdf2 = gdf[[field, "geometry"]].copy()
    gdf2[field] = gdf2[field].astype(str)
    gdf2 = gdf2.dissolve(by=field)

    names = [str(k) for k in gdf2.index.tolist()]
    polys = [geom for geom in gdf2.geometry.values]

    regions = regionmask.Regions(polys, names=names, abbrevs=names)
    mask = regions.mask(lon=ref["lon"].values, lat=ref["lat"].values)
    climate_id = (mask + 1).astype("float32")  # 1..N

    mapping = {i + 1: names[i] for i in range(len(names))}
    climate_id = xr.DataArray(climate_id, coords={"lat": ref["lat"].values, "lon": ref["lon"].values}, dims=("lat", "lon"))
    return climate_id, mapping


def load_all_basic(cfg: dict):
    """
    Shared loader for (ref, products, climate mask) using cfg["io"] and cfg["preprocess"].

    Expected schema:
      cfg["io"]["ref_path"], optional cfg["io"]["ref_var"]
      cfg["io"]["products"] -> list of {name, path, var?}
      cfg["io"]["china_shp"], cfg["io"]["climate_field"]
      cfg["preprocess"]["time_floor"], ["chunks"], ["interp_method"]
    """
    io = cfg["io"]
    pp = cfg.get("preprocess", {}) or {}
    chunks = pp.get("chunks", None)
    floor = pp.get("time_floor", "D")
    interp = pp.get("interp_method", "linear")

    ref = open_da(io["ref_path"], io.get("ref_var", None), chunks=chunks)
    ref = standardize_time(ref, floor=floor)

    prods = []
    for p in io["products"]:
        da = open_da(p["path"], p.get("var", None), chunks=chunks)
        da = standardize_time(da, floor=floor)
        da = align_to_ref_grid(da, ref, method=interp)
        prods.append((p["name"], da))

    # time intersection
    idxs = [pd.DatetimeIndex(ref["time"].values)]
    idxs += [pd.DatetimeIndex(da["time"].values) for _, da in prods]
    common = intersect_datetimes(*idxs)
    if len(common) == 0:
        raise ValueError("time intersection is empty after standardize_time; check time axes and ranges.")

    ref = ref.sel(time=common)
    prods = [(n, da.sel(time=common)) for n, da in prods]

    climate_id, mapping = build_climate_id_mask(io["china_shp"], io["climate_field"], ref)
    return ref, prods, climate_id, mapping
