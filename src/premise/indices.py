# -*- coding: utf-8 -*-

"""
indices
=======

Hydroclimatic indices:

- SPI  : Standardized Precipitation Index
- SPEI : Standardized Precipitation–Evapotranspiration Index (simplified)
- SRI  : Streamflow Drought Index
- STI  : Standardized Temperature Index

This module provides simplified but scalable implementations based on xarray.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from scipy import stats


def _aggregate_to_monthly(
    da: xr.DataArray,
    scale: int = 1,
    freq: str = "M",
    op: Literal["sum", "mean"] = "sum",
    is_monthly_input: bool = False,
) -> xr.DataArray:
    """
    Aggregate a time series to monthly resolution and apply an n-month
    rolling sum or mean.
    """
    if not is_monthly_input:
        if op == "sum":
            monthly = da.resample(time=freq).sum(skipna=True)
        else:
            monthly = da.resample(time=freq).mean(skipna=True)
    else:
        monthly = da

    if scale > 1:
        if op == "sum":
            agg = monthly.rolling(time=scale, min_periods=scale).sum()
        else:
            agg = monthly.rolling(time=scale, min_periods=scale).mean()
    else:
        agg = monthly

    return agg


def _spi_like_1d(
    series: np.ndarray,
    min_obs: int = 10,
    dist: Literal["gamma", "normal"] = "gamma",
) -> np.ndarray:
    """
    Core 1D SPI/SRI-like index computation.
    """
    series = np.asarray(series, dtype=float)
    out = np.full_like(series, np.nan, dtype=float)

    mask = np.isfinite(series)
    if mask.sum() < min_obs:
        return out

    data = series[mask]

    if dist == "gamma":
        data_pos = data[data > 0]
        if data_pos.size < min_obs:
            return out

        shape, loc, scale = stats.gamma.fit(data_pos, floc=0.0)

        series_pos = np.where(series <= 0, np.nan, series)
        cdf = np.full(series.shape, np.nan, dtype=float)
        valid_pos = np.isfinite(series_pos)
        cdf[valid_pos] = stats.gamma.cdf(
            series_pos[valid_pos],
            shape,
            loc=loc,
            scale=scale,
        )
    elif dist == "normal":
        mu = data.mean()
        sigma = data.std(ddof=1)
        if sigma == 0 or not np.isfinite(sigma):
            return out
        cdf = np.full(series.shape, np.nan, dtype=float)
        cdf[mask] = stats.norm.cdf(series[mask], loc=mu, scale=sigma)
    else:
        raise ValueError(f"Unsupported dist: {dist}")

    cdf = np.clip(cdf, 1e-6, 1.0 - 1e-6)
    spi = stats.norm.ppf(cdf)
    out[mask] = spi[mask]
    return out


def _zscore_1d(
    series: np.ndarray,
    min_obs: int = 10,
) -> np.ndarray:
    """
    1D z-score standardisation.
    """
    series = np.asarray(series, dtype=float)
    out = np.full_like(series, np.nan, dtype=float)

    mask = np.isfinite(series)
    if mask.sum() < min_obs:
        return out

    data = series[mask]
    mu = data.mean()
    sigma = data.std(ddof=1)
    if sigma == 0 or not np.isfinite(sigma):
        return out

    out[mask] = (series[mask] - mu) / sigma
    return out


def calc_spi(
    precip: xr.DataArray,
    scale: int = 3,
    *,
    freq: str = "M",
    is_monthly_input: bool = False,
    min_obs: int = 10,
) -> xr.DataArray:
    """
    Compute the Standardized Precipitation Index (SPI).
    """
    agg = _aggregate_to_monthly(
        precip,
        scale=scale,
        freq=freq,
        op="sum",
        is_monthly_input=is_monthly_input,
    )

    spi = xr.apply_ufunc(
        _spi_like_1d,
        agg,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"min_obs": min_obs, "dist": "gamma"},
        output_dtypes=[float],
    )
    spi.name = f"SPI_{scale}"
    return spi


def calc_sri(
    runoff: xr.DataArray,
    scale: int = 3,
    *,
    freq: str = "M",
    is_monthly_input: bool = False,
    min_obs: int = 10,
) -> xr.DataArray:
    """
    Compute the Streamflow Drought Index (SRI).
    """
    agg = _aggregate_to_monthly(
        runoff,
        scale=scale,
        freq=freq,
        op="sum",
        is_monthly_input=is_monthly_input,
    )

    sri = xr.apply_ufunc(
        _spi_like_1d,
        agg,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"min_obs": min_obs, "dist": "gamma"},
        output_dtypes=[float],
    )
    sri.name = f"SRI_{scale}"
    return sri


def calc_spei(
    precip: xr.DataArray,
    pet: xr.DataArray,
    scale: int = 3,
    *,
    freq: str = "M",
    is_monthly_input: bool = False,
    min_obs: int = 10,
) -> xr.DataArray:
    """
    Compute a simplified SPEI:

    - D = P - PET
    - Aggregate to n-month totals
    - z-score standardisation
    """
    precip, pet = xr.align(precip, pet, join="inner")
    balance = precip - pet

    agg = _aggregate_to_monthly(
        balance,
        scale=scale,
        freq=freq,
        op="sum",
        is_monthly_input=is_monthly_input,
    )

    spei = xr.apply_ufunc(
        _zscore_1d,
        agg,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"min_obs": min_obs},
        output_dtypes=[float],
    )
    spei.name = f"SPEI_{scale}"
    return spei


def calc_sti(
    temp: xr.DataArray,
    scale: int = 1,
    *,
    freq: str = "M",
    is_monthly_input: bool = False,
    min_obs: int = 10,
) -> xr.DataArray:
    """
    Compute a Standardized Temperature Index (STI):

    - Aggregate temperature in n-month windows (mean)
    - z-score standardisation
    """
    agg = _aggregate_to_monthly(
        temp,
        scale=scale,
        freq=freq,
        op="mean",
        is_monthly_input=is_monthly_input,
    )

    sti = xr.apply_ufunc(
        _zscore_1d,
        agg,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"min_obs": min_obs},
        output_dtypes=[float],
    )
    sti.name = f"STI_{scale}"
    return sti
