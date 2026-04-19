# -*- coding: utf-8 -*-
"""Continuous and event-detection metrics for precipitation evaluation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr

__all__ = [
    "bias",
    "mae",
    "mse",
    "rmse",
    "corr",
    "kge",
    "nse",
    "pod",
    "far",
    "csi",
    "fbias",
    "hss",
]


def _ensure_array(a) -> np.ndarray:
    if isinstance(a, xr.DataArray):
        a = a.values
    return np.asarray(a).ravel()


def _valid_pair(obs, sim) -> Tuple[np.ndarray, np.ndarray]:
    obs_arr = _ensure_array(obs)
    sim_arr = _ensure_array(sim)
    mask = np.isfinite(obs_arr) & np.isfinite(sim_arr)
    return obs_arr[mask], sim_arr[mask]


def bias(obs, sim) -> float:
    o, s = _valid_pair(obs, sim)
    return float(np.nan) if o.size == 0 else float(np.mean(s - o))


def mae(obs, sim) -> float:
    o, s = _valid_pair(obs, sim)
    return float(np.nan) if o.size == 0 else float(np.mean(np.abs(s - o)))


def mse(obs, sim) -> float:
    o, s = _valid_pair(obs, sim)
    return float(np.nan) if o.size == 0 else float(np.mean((s - o) ** 2))


def rmse(obs, sim) -> float:
    val = mse(obs, sim)
    return float(np.nan) if np.isnan(val) else float(np.sqrt(val))


def corr(obs, sim) -> float:
    o, s = _valid_pair(obs, sim)
    if o.size < 2:
        return float(np.nan)
    return float(np.corrcoef(o, s)[0, 1])


def kge(obs, sim) -> float:
    o, s = _valid_pair(obs, sim)
    if o.size < 2:
        return float(np.nan)
    r = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s, ddof=1) / np.std(o, ddof=1) if np.std(o, ddof=1) != 0 else np.nan
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def nse(obs, sim) -> float:
    o, s = _valid_pair(obs, sim)
    if o.size < 2:
        return float(np.nan)
    denom = np.sum((o - np.mean(o)) ** 2)
    if denom == 0:
        return float(np.nan)
    return float(1.0 - np.sum((s - o) ** 2) / denom)


def contingency_table(obs, sim, threshold: float = 1.0) -> Tuple[int, int, int, int]:
    o, s = _valid_pair(obs, sim)
    if o.size == 0:
        return 0, 0, 0, 0
    obs_event = o >= threshold
    sim_event = s >= threshold
    hits = int(np.sum(obs_event & sim_event))
    misses = int(np.sum(obs_event & (~sim_event)))
    false_alarms = int(np.sum((~obs_event) & sim_event))
    correct_negatives = int(np.sum((~obs_event) & (~sim_event)))
    return hits, misses, false_alarms, correct_negatives


def pod(obs, sim, threshold: float = 1.0) -> float:
    h, m, _, _ = contingency_table(obs, sim, threshold)
    return float(np.nan) if (h + m) == 0 else float(h / (h + m))


def far(obs, sim, threshold: float = 1.0) -> float:
    h, _, f, _ = contingency_table(obs, sim, threshold)
    return float(np.nan) if (h + f) == 0 else float(f / (h + f))


def csi(obs, sim, threshold: float = 1.0) -> float:
    h, m, f, _ = contingency_table(obs, sim, threshold)
    denom = h + m + f
    return float(np.nan) if denom == 0 else float(h / denom)


def fbias(obs, sim, threshold: float = 1.0) -> float:
    h, m, f, _ = contingency_table(obs, sim, threshold)
    return float(np.nan) if (h + m) == 0 else float((h + f) / (h + m))


def hss(obs, sim, threshold: float = 1.0) -> float:
    h, m, f, c = contingency_table(obs, sim, threshold)
    numerator = 2 * (h * c - m * f)
    denominator = (h + m) * (m + c) + (h + f) * (f + c)
    return float(np.nan) if denominator == 0 else float(numerator / denominator)
