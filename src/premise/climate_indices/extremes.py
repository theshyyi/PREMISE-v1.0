# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import xarray as xr


def _get_time_dim(da: xr.DataArray, time_dim: str) -> str:
    if time_dim not in da.dims:
        raise KeyError(f"time_dim '{time_dim}' not found in DataArray dims.")
    return time_dim


def rx1day(pr: xr.DataArray, *, freq: str = "A", time_dim: str = "time") -> xr.DataArray:
    time_dim = _get_time_dim(pr, time_dim)
    out = pr.resample({time_dim: freq}).max(skipna=True)
    out.name = "Rx1day"
    return out


def rx5day(pr: xr.DataArray, *, freq: str = "A", time_dim: str = "time", window: int = 5) -> xr.DataArray:
    time_dim = _get_time_dim(pr, time_dim)
    pr_5 = pr.rolling({time_dim: window}, min_periods=window).sum()
    out = pr_5.resample({time_dim: freq}).max(skipna=True)
    out.name = "Rx5day"
    return out


def prcptot(pr: xr.DataArray, *, wet_threshold: float = 1.0, freq: str = "A", time_dim: str = "time") -> xr.DataArray:
    time_dim = _get_time_dim(pr, time_dim)
    wet = pr.where(pr >= wet_threshold)
    out = wet.resample({time_dim: freq}).sum(skipna=True)
    out.name = "PRCPTOT"
    return out


def sdii(pr: xr.DataArray, *, wet_threshold: float = 1.0, freq: str = "A", time_dim: str = "time") -> xr.DataArray:
    time_dim = _get_time_dim(pr, time_dim)
    wet = pr.where(pr >= wet_threshold)
    tot = wet.resample({time_dim: freq}).sum(skipna=True)
    wet_days = wet.resample({time_dim: freq}).count()
    out = tot / wet_days
    out.name = "SDII"
    return out


def r10mm(pr: xr.DataArray, *, threshold: float = 10.0, freq: str = "A", time_dim: str = "time") -> xr.DataArray:
    time_dim = _get_time_dim(pr, time_dim)
    out = (pr >= threshold).resample({time_dim: freq}).sum()
    out.name = "R10mm"
    return out


def r20mm(pr: xr.DataArray, *, threshold: float = 20.0, freq: str = "A", time_dim: str = "time") -> xr.DataArray:
    time_dim = _get_time_dim(pr, time_dim)
    out = (pr >= threshold).resample({time_dim: freq}).sum()
    out.name = "R20mm"
    return out


def _percentile_threshold(pr: xr.DataArray, q: float, *, time_dim: str = "time", base_period: Optional[Tuple[str, str]] = None, wet_threshold: Optional[float] = None) -> xr.DataArray:
    time_dim = _get_time_dim(pr, time_dim)
    pr_base = pr.sel({time_dim: slice(*base_period)}) if base_period is not None else pr
    if wet_threshold is not None:
        pr_base = pr_base.where(pr_base >= wet_threshold)
    return pr_base.quantile(q / 100.0, dim=time_dim, keep_attrs=True)


def r95p(pr: xr.DataArray, *, time_dim: str = "time", base_period: Optional[Tuple[str, str]] = None, wet_threshold: Optional[float] = 1.0, freq: str = "A") -> xr.DataArray:
    time_dim = _get_time_dim(pr, time_dim)
    thresh = _percentile_threshold(pr, 95.0, time_dim=time_dim, base_period=base_period, wet_threshold=wet_threshold)
    out = pr.where(pr > thresh).resample({time_dim: freq}).sum(skipna=True)
    out.name = "R95p"
    return out


def r99p(pr: xr.DataArray, *, time_dim: str = "time", base_period: Optional[Tuple[str, str]] = None, wet_threshold: Optional[float] = 1.0, freq: str = "A") -> xr.DataArray:
    time_dim = _get_time_dim(pr, time_dim)
    thresh = _percentile_threshold(pr, 99.0, time_dim=time_dim, base_period=base_period, wet_threshold=wet_threshold)
    out = pr.where(pr > thresh).resample({time_dim: freq}).sum(skipna=True)
    out.name = "R99p"
    return out


def _max_run_length_1d(series: np.ndarray, is_event: np.ndarray) -> int:
    out = 0
    run = 0
    for flag in is_event:
        if flag:
            run += 1
            if run > out:
                out = run
        else:
            run = 0
    return out


def _cdd_1d(series: np.ndarray, dry_threshold: float) -> float:
    series = np.asarray(series, dtype=float)
    mask = np.isfinite(series)
    if mask.sum() == 0:
        return np.nan
    cond = np.zeros_like(series, dtype=bool)
    cond[mask] = series[mask] < dry_threshold
    return float(_max_run_length_1d(series, cond))


def _cwd_1d(series: np.ndarray, wet_threshold: float) -> float:
    series = np.asarray(series, dtype=float)
    mask = np.isfinite(series)
    if mask.sum() == 0:
        return np.nan
    cond = np.zeros_like(series, dtype=bool)
    cond[mask] = series[mask] >= wet_threshold
    return float(_max_run_length_1d(series, cond))


def cdd(pr: xr.DataArray, *, dry_threshold: float = 1.0, time_dim: str = "time") -> xr.DataArray:
    time_dim = _get_time_dim(pr, time_dim)
    def _per_year(group: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            _cdd_1d, group,
            input_core_dims=[[time_dim]], output_core_dims=[[]], vectorize=True,
            dask="parallelized", kwargs={"dry_threshold": dry_threshold}, output_dtypes=[float],
        )
    out = pr.groupby(f"{time_dim}.year").map(_per_year).rename("CDD").rename({"year": time_dim})
    return out


def cwd(pr: xr.DataArray, *, wet_threshold: float = 1.0, time_dim: str = "time") -> xr.DataArray:
    time_dim = _get_time_dim(pr, time_dim)
    def _per_year(group: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            _cwd_1d, group,
            input_core_dims=[[time_dim]], output_core_dims=[[]], vectorize=True,
            dask="parallelized", kwargs={"wet_threshold": wet_threshold}, output_dtypes=[float],
        )
    out = pr.groupby(f"{time_dim}.year").map(_per_year).rename("CWD").rename({"year": time_dim})
    return out
