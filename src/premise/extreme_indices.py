# -*- coding: utf-8 -*-

"""
extreme_indices
===============

自实现的一组常用极端降水指数（ETCCDI-like），依赖 numpy + xarray：

- Rx1day  : max 1-day precipitation
- Rx5day  : max 5-day precipitation
- PRCPTOT : annual/seasonal total wet-day precipitation
- SDII    : Simple Daily Intensity Index
- R10mm   : number of days with RR >= 10 mm
- R20mm   : number of days with RR >= 20 mm
- R95p    : total precipitation from days above 95th percentile
- R99p    : total precipitation from days above 99th percentile
- CDD     : maximum length of dry spell (RR < dry_threshold)
- CWD     : maximum length of wet spell (RR >= wet_threshold)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import xarray as xr

_all_ = [
    "rx1day",
    "rx5day",
    "prcptot",
    "sdii",
    "r10mm",
    "r20mm",
    "r95p",
    "r99p",
    "cdd",
    "cwd"
]



def _get_time_dim(da: xr.DataArray, time_dim: str) -> str:
    if time_dim not in da.dims:
        raise KeyError(f"time_dim '{time_dim}' not found in DataArray dims.")
    return time_dim


# ========== 简单 resample 型指数 ==========

def rx1day(
    pr: xr.DataArray,
    *,
    freq: str = "A",
    time_dim: str = "time",
) -> xr.DataArray:
    """
    Rx1day: 最大逐日降水量。

    freq:
        "A"  年度
        "Q-DEC" 季度 (DJF/MAM/JJA/SON)
        也可以用 "YS" 等更细化频率
    """
    time_dim = _get_time_dim(pr, time_dim)
    rx1 = pr.resample({time_dim: freq}).max(skipna=True)
    rx1.name = "Rx1day"
    return rx1


def rx5day(
    pr: xr.DataArray,
    *,
    freq: str = "A",
    time_dim: str = "time",
    window: int = 5,
) -> xr.DataArray:
    """
    Rx5day: 最大 5 日累计降水量。
    """
    time_dim = _get_time_dim(pr, time_dim)
    pr_5 = pr.rolling({time_dim: window}, min_periods=window).sum()
    rx5 = pr_5.resample({time_dim: freq}).max(skipna=True)
    rx5.name = "Rx5day"
    return rx5


def prcptot(
    pr: xr.DataArray,
    *,
    wet_threshold: float = 1.0,
    freq: str = "A",
    time_dim: str = "time",
) -> xr.DataArray:
    """
    PRCPTOT: 总湿日降水量（RR >= wet_threshold 的日降水之和）。
    """
    time_dim = _get_time_dim(pr, time_dim)
    wet = pr.where(pr >= wet_threshold)
    tot = wet.resample({time_dim: freq}).sum(skipna=True)
    tot.name = "PRCPTOT"
    return tot


def sdii(
    pr: xr.DataArray,
    *,
    wet_threshold: float = 1.0,
    freq: str = "A",
    time_dim: str = "time",
) -> xr.DataArray:
    """
    SDII: simple daily intensity index.

    定义：PRCPTOT / 湿日数 (RR >= wet_threshold)。
    """
    time_dim = _get_time_dim(pr, time_dim)
    wet = pr.where(pr >= wet_threshold)

    tot = wet.resample({time_dim: freq}).sum(skipna=True)
    wet_days = wet.resample({time_dim: freq}).count()

    sdii_val = tot / wet_days
    sdii_val.name = "SDII"
    return sdii_val


def r10mm(
    pr: xr.DataArray,
    *,
    threshold: float = 10.0,
    freq: str = "A",
    time_dim: str = "time",
) -> xr.DataArray:
    """
    R10mm: RR >= 10 mm 的日数（可指定 threshold）。
    """
    time_dim = _get_time_dim(pr, time_dim)
    mask = pr >= threshold
    r10 = mask.resample({time_dim: freq}).sum()
    r10.name = "R10mm"
    return r10


def r20mm(
    pr: xr.DataArray,
    *,
    threshold: float = 20.0,
    freq: str = "A",
    time_dim: str = "time",
) -> xr.DataArray:
    """
    R20mm: RR >= 20 mm 的日数（可指定 threshold）。
    """
    time_dim = _get_time_dim(pr, time_dim)
    mask = pr >= threshold
    r20 = mask.resample({time_dim: freq}).sum()
    r20.name = "R20mm"
    return r20


# ========== 百分位型指数 (R95p / R99p) ==========

def _percentile_threshold(
    pr: xr.DataArray,
    q: float,
    *,
    time_dim: str = "time",
    base_period: Optional[Tuple[str, str]] = None,
    wet_threshold: Optional[float] = None,
) -> xr.DataArray:
    """
    计算每个格点上的百分位数阈值（可选基准期和湿日阈值）。
    """
    time_dim = _get_time_dim(pr, time_dim)

    if base_period is not None:
        start, end = base_period
        pr_base = pr.sel({time_dim: slice(start, end)})
    else:
        pr_base = pr

    if wet_threshold is not None:
        pr_base = pr_base.where(pr_base >= wet_threshold)

    thresh = pr_base.quantile(q / 100.0, dim=time_dim, keep_attrs=True)
    return thresh


def r95p(
    pr: xr.DataArray,
    *,
    time_dim: str = "time",
    base_period: Optional[Tuple[str, str]] = None,
    wet_threshold: Optional[float] = 1.0,
    freq: str = "A",
) -> xr.DataArray:
    """
    R95pTOT (简化版)：超过基准期 95 百分位的日降水量之和。
    """
    time_dim = _get_time_dim(pr, time_dim)
    thresh = _percentile_threshold(
        pr,
        95.0,
        time_dim=time_dim,
        base_period=base_period,
        wet_threshold=wet_threshold,
    )

    # 广播到 time 维度
    excess = pr.where(pr > thresh)

    r95 = excess.resample({time_dim: freq}).sum(skipna=True)
    r95.name = "R95p"
    return r95


def r99p(
    pr: xr.DataArray,
    *,
    time_dim: str = "time",
    base_period: Optional[Tuple[str, str]] = None,
    wet_threshold: Optional[float] = 1.0,
    freq: str = "A",
) -> xr.DataArray:
    """
    R99pTOT (简化版)：超过基准期 99 百分位的日降水量之和。
    """
    time_dim = _get_time_dim(pr, time_dim)
    thresh = _percentile_threshold(
        pr,
        99.0,
        time_dim=time_dim,
        base_period=base_period,
        wet_threshold=wet_threshold,
    )
    excess = pr.where(pr > thresh)

    r99 = excess.resample({time_dim: freq}).sum(skipna=True)
    r99.name = "R99p"
    return r99


# ========== CDD / CWD（最长连续干/湿日） ==========

def _max_run_length_1d(series: np.ndarray, is_event: np.ndarray) -> int:
    """
    计算 1D 序列中布尔条件为 True 的最长连续长度。
    """
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


def cdd(
    pr: xr.DataArray,
    *,
    dry_threshold: float = 1.0,
    time_dim: str = "time",
) -> xr.DataArray:
    """
    CDD: 每年（或每季）最长连续干日数（RR < dry_threshold）。

    这里按年分组：time.year
    """
    time_dim = _get_time_dim(pr, time_dim)

    def _per_year(group: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            _cdd_1d,
            group,
            input_core_dims=[[time_dim]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            kwargs={"dry_threshold": dry_threshold},
            output_dtypes=[float],
        )

    cdd_year = pr.groupby(f"{time_dim}.year").map(_per_year)
    cdd_year = cdd_year.rename("CDD")
    cdd_year = cdd_year.rename({"year": time_dim})
    return cdd_year


def cwd(
    pr: xr.DataArray,
    *,
    wet_threshold: float = 1.0,
    time_dim: str = "time",
) -> xr.DataArray:
    """
    CWD: 每年（或每季）最长连续湿日数（RR >= wet_threshold）。

    这里按年分组：time.year
    """
    time_dim = _get_time_dim(pr, time_dim)

    def _per_year(group: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            _cwd_1d,
            group,
            input_core_dims=[[time_dim]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            kwargs={"wet_threshold": wet_threshold},
            output_dtypes=[float],
        )

    cwd_year = pr.groupby(f"{time_dim}.year").map(_per_year)
    cwd_year = cwd_year.rename("CWD")
    cwd_year = cwd_year.rename({"year": time_dim})
    return cwd_year
