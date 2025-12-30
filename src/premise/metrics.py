# -*- coding: utf-8 -*-

"""
metrics
=======

Continuous and categorical evaluation metrics:

- Continuous: BIAS, MAE, MSE, RMSE, correlation, KGE
- Detection-based: POD, FAR, CSI, FBIAS via 2x2 contingency table
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr

_all_ = [
    "bias",
    "mae",
    "mse",
    "rmse",
    "corr",
    "kge",
    "pod",
    "far",
    "csi",
    "fbias",
    "hss"
]
def _ensure_array(a) -> np.ndarray:
    """
    Convert input to a 1D numpy array.
    """
    if isinstance(a, xr.DataArray):
        a = a.values
    arr = np.asarray(a)
    return arr.ravel()


def _valid_pair(obs, sim) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter finite-value pairs from obs and sim.
    """
    obs_arr = _ensure_array(obs)
    sim_arr = _ensure_array(sim)
    mask = np.isfinite(obs_arr) & np.isfinite(sim_arr)
    return obs_arr[mask], sim_arr[mask]


# ========== Continuous metrics ==========

def bias(obs, sim) -> float:
    """
    Mean bias = mean(sim - obs)
    """
    o, s = _valid_pair(obs, sim)
    if o.size == 0:
        return np.nan
    return float(np.mean(s - o))


def mae(obs, sim) -> float:
    """
    Mean absolute error.
    """
    o, s = _valid_pair(obs, sim)
    if o.size == 0:
        return np.nan
    return float(np.mean(np.abs(s - o)))


def mse(obs, sim) -> float:
    """
    Mean squared error.
    """
    o, s = _valid_pair(obs, sim)
    if o.size == 0:
        return np.nan
    return float(np.mean((s - o) ** 2))


def rmse(obs, sim) -> float:
    """
    Root mean squared error.
    """
    val = mse(obs, sim)
    return float(np.sqrt(val)) if np.isfinite(val) else np.nan


def corr(obs, sim) -> float:
    """
    Pearson correlation coefficient.
    """
    o, s = _valid_pair(obs, sim)
    if o.size < 2:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])


def kge(obs, sim) -> float:
    """
    Kling–Gupta efficiency (KGE).
    """
    o, s = _valid_pair(obs, sim)
    if o.size < 2:
        return np.nan

    r = np.corrcoef(o, s)[0, 1]

    mean_o = np.mean(o)
    mean_s = np.mean(s)
    std_o = np.std(o, ddof=1)
    std_s = np.std(s, ddof=1)

    if std_o == 0 or mean_o == 0:
        return np.nan

    alpha = std_s / std_o
    beta = mean_s / mean_o

    kge_val = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(kge_val)


# ========== 2x2 contingency table and detection scores ==========

def contingency_table(
    obs,
    sim,
    threshold: float,
) -> Tuple[int, int, int, int]:
    """
    2x2 contingency table:

        H: hits (obs>=th, sim>=th)
        M: misses (obs>=th, sim<th)
        F: false alarms (obs<th, sim>=th)
        C: correct negatives (obs<th, sim<th)
    """
    o, s = _valid_pair(obs, sim)
    if o.size == 0:
        return 0, 0, 0, 0

    flag_o = o >= threshold
    flag_s = s >= threshold

    H = int(np.sum(flag_o & flag_s))
    M = int(np.sum(flag_o & ~flag_s))
    F = int(np.sum(~flag_o & flag_s))
    C = int(np.sum(~flag_o & ~flag_s))

    return H, M, F, C


def pod(obs, sim, threshold: float) -> float:
    """
    Probability of detection (POD) = H / (H+M)
    """
    H, M, F, C = contingency_table(obs, sim, threshold)
    return H / (H + M) if (H + M) > 0 else np.nan


def far(obs, sim, threshold: float) -> float:
    """
    False alarm ratio (FAR) = F / (H+F)
    """
    H, M, F, C = contingency_table(obs, sim, threshold)
    return F / (H + F) if (H + F) > 0 else np.nan


def csi(obs, sim, threshold: float) -> float:
    """
    Critical success index (CSI) = H / (H+M+F)
    """
    H, M, F, C = contingency_table(obs, sim, threshold)
    denom = H + M + F
    return H / denom if denom > 0 else np.nan


def fbias(obs, sim, threshold: float) -> float:
    """
    Frequency bias (FBIAS) = (H+F) / (H+M)
    """
    H, M, F, C = contingency_table(obs, sim, threshold)
    denom = H + M
    return (H + F) / denom if denom > 0 else np.nan


def hss(obs, sim, threshold: float) -> float:
    """
    Frequency bias (HSS) = 2(HC-FM) / (H+M)(M+C)+(H+F)(F+C)
    """
    H, M, F, C = contingency_table(obs, sim, threshold)
    denom1 = H * C - F * M
    denom2 = H + M
    demon3 = M + C
    demon4 = H + F
    demon5 = F + C
    return denom1 * 2 / (denom2 * demon3 + demon4 * demon5) if (denom2 * demon3 + demon4 * demon5) != 0 else np.nan