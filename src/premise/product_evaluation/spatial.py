
from __future__ import annotations

from typing import Callable

import numpy as np
import xarray as xr

from .metrics import contingency_table


def _valid_pair_1d(obs: np.ndarray, sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


def _apply_1d_stat(obs: xr.DataArray, sim: xr.DataArray, func: Callable[[np.ndarray, np.ndarray], float]) -> xr.DataArray:
    return xr.apply_ufunc(
        lambda o, s: func(*_valid_pair_1d(o, s)),
        obs,
        sim,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


def bias_map(obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray:
    out = (sim - obs).mean("time", skipna=True)
    out.name = "BIAS"
    return out


def mae_map(obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray:
    out = np.abs(sim - obs).mean("time", skipna=True)
    out.name = "MAE"
    return out


def rmse_map(obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray:
    out = np.sqrt(((sim - obs) ** 2).mean("time", skipna=True))
    out.name = "RMSE"
    return out


def corr_map(obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray:
    out = xr.corr(sim, obs, dim="time")
    out.name = "CORR"
    return out


def nse_map(obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray:
    def _nse(o: np.ndarray, s: np.ndarray) -> float:
        o, s = _valid_pair_1d(o, s)
        if o.size < 2:
            return float(np.nan)
        denom = np.sum((o - np.mean(o)) ** 2)
        if denom == 0:
            return float(np.nan)
        return float(1.0 - np.sum((s - o) ** 2) / denom)
    out = _apply_1d_stat(obs, sim, _nse)
    out.name = "NSE"
    return out


def kge_map(obs: xr.DataArray, sim: xr.DataArray) -> xr.DataArray:
    def _kge(o: np.ndarray, s: np.ndarray) -> float:
        o, s = _valid_pair_1d(o, s)
        if o.size < 2:
            return float(np.nan)
        r = np.corrcoef(o, s)[0, 1]
        ostd = np.std(o, ddof=1)
        if ostd == 0:
            return float(np.nan)
        alpha = np.std(s, ddof=1) / ostd
        omean = np.mean(o)
        if omean == 0:
            return float(np.nan)
        beta = np.mean(s) / omean
        return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    out = _apply_1d_stat(obs, sim, _kge)
    out.name = "KGE"
    return out


def _event_metric_map(obs: xr.DataArray, sim: xr.DataArray, threshold: float, kind: str) -> xr.DataArray:
    def _calc(o: np.ndarray, s: np.ndarray) -> float:
        h, m, f, c = contingency_table(o, s, threshold)
        if kind == "POD":
            return np.nan if (h + m) == 0 else h / (h + m)
        if kind == "FAR":
            return np.nan if (h + f) == 0 else f / (h + f)
        if kind == "CSI":
            denom = h + m + f
            return np.nan if denom == 0 else h / denom
        if kind == "FBIAS":
            return np.nan if (h + m) == 0 else (h + f) / (h + m)
        if kind == "HSS":
            denom = (h + m) * (m + c) + (h + f) * (f + c)
            return np.nan if denom == 0 else 2 * (h * c - m * f) / denom
        raise ValueError(kind)

    out = xr.apply_ufunc(
        _calc,
        obs,
        sim,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        kwargs={"threshold": threshold, "kind": kind},
        output_dtypes=[float],
    )
    out.name = kind
    return out


def pod_map(obs: xr.DataArray, sim: xr.DataArray, threshold: float = 1.0) -> xr.DataArray:
    return _event_metric_map(obs, sim, threshold, "POD")


def far_map(obs: xr.DataArray, sim: xr.DataArray, threshold: float = 1.0) -> xr.DataArray:
    return _event_metric_map(obs, sim, threshold, "FAR")


def csi_map(obs: xr.DataArray, sim: xr.DataArray, threshold: float = 1.0) -> xr.DataArray:
    return _event_metric_map(obs, sim, threshold, "CSI")


def fbias_map(obs: xr.DataArray, sim: xr.DataArray, threshold: float = 1.0) -> xr.DataArray:
    return _event_metric_map(obs, sim, threshold, "FBIAS")


def hss_map(obs: xr.DataArray, sim: xr.DataArray, threshold: float = 1.0) -> xr.DataArray:
    return _event_metric_map(obs, sim, threshold, "HSS")
