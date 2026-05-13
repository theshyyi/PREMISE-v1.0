from __future__ import annotations

import numpy as np
import pandas as pd

from .normalize import normalize_to_benefit_scale


def equal_weights(metrics: list[str]) -> pd.Series:
    if len(metrics) == 0:
        raise ValueError('metrics cannot be empty')
    w = np.full(len(metrics), 1.0 / len(metrics), dtype=float)
    return pd.Series(w, index=metrics, name='weight')


def critic_weights(metric_df: pd.DataFrame, metric_directions: dict[str, str]) -> pd.Series:
    Z = normalize_to_benefit_scale(metric_df, metric_directions, method='minmax')
    std = Z.std(axis=0, ddof=1).fillna(0.0)
    corr = Z.corr().fillna(0.0)
    info = {}
    for j in Z.columns:
        conflict = (1.0 - corr.loc[j]).sum()
        info[j] = float(std[j] * conflict)
    s = pd.Series(info, dtype=float)
    total = float(s.sum())
    if total <= 0 or not np.isfinite(total):
        return equal_weights(list(metric_df.columns))
    return (s / total).rename('weight')


def entropy_weights(metric_df: pd.DataFrame, metric_directions: dict[str, str], eps: float = 1e-12) -> pd.Series:
    Z = normalize_to_benefit_scale(metric_df, metric_directions, method='minmax').fillna(0.0)
    X = Z.to_numpy(dtype=float)
    # shift in case of all zeros per column
    col_sums = X.sum(axis=0)
    n, m = X.shape
    P = np.zeros_like(X, dtype=float)
    for j in range(m):
        if col_sums[j] <= 0:
            P[:, j] = 1.0 / max(n, 1)
        else:
            P[:, j] = X[:, j] / col_sums[j]
    k = 1.0 / np.log(max(n, 2))
    E = -k * np.sum(P * np.log(P + eps), axis=0)
    d = 1.0 - E
    if np.sum(d) <= 0 or not np.isfinite(np.sum(d)):
        return equal_weights(list(metric_df.columns))
    w = d / np.sum(d)
    return pd.Series(w, index=metric_df.columns, name='weight')
