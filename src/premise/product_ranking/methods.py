from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .normalize import normalize_to_benefit_scale, prepare_numeric_matrix, standardize_directions
from .weights import equal_weights


def _rank_series(values: pd.Series, *, ascending: bool) -> pd.Series:
    return values.rank(method='average', ascending=ascending)


def simple_metric_ranks(df: pd.DataFrame, product_col: str, metric_directions: Mapping[str, str]) -> pd.DataFrame:
    dirs = standardize_directions(metric_directions)
    metrics = list(dirs.keys())
    X = prepare_numeric_matrix(df, metrics)
    out = pd.DataFrame({product_col: df[product_col].values})
    for m in metrics:
        out[f'{m}_rank'] = _rank_series(X[m], ascending=(dirs[m] == 'cost'))
    return out


def average_rank(df: pd.DataFrame, product_col: str, metric_directions: Mapping[str, str]) -> pd.DataFrame:
    rank_df = simple_metric_ranks(df, product_col, metric_directions)
    rank_cols = [c for c in rank_df.columns if c.endswith('_rank')]
    out = rank_df[[product_col]].copy()
    out['average_rank_score'] = rank_df[rank_cols].mean(axis=1)
    out['rank'] = out['average_rank_score'].rank(method='average', ascending=True)
    return out.sort_values(['rank', 'average_rank_score', product_col]).reset_index(drop=True)


def borda_rank(df: pd.DataFrame, product_col: str, metric_directions: Mapping[str, str]) -> pd.DataFrame:
    rank_df = simple_metric_ranks(df, product_col, metric_directions)
    rank_cols = [c for c in rank_df.columns if c.endswith('_rank')]
    n = len(rank_df)
    points = pd.DataFrame(index=rank_df.index)
    for c in rank_cols:
        points[c.replace('_rank', '_points')] = n + 1 - rank_df[c]
    out = rank_df[[product_col]].copy()
    out['borda_score'] = points.sum(axis=1)
    out['rank'] = out['borda_score'].rank(method='average', ascending=False)
    return out.sort_values(['rank', 'borda_score'], ascending=[True, False]).reset_index(drop=True)


def weighted_sum_rank(
    df: pd.DataFrame,
    product_col: str,
    metric_directions: Mapping[str, str],
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    dirs = standardize_directions(metric_directions)
    metrics = list(dirs.keys())
    X = prepare_numeric_matrix(df, metrics)
    Z = normalize_to_benefit_scale(X, dirs, method='minmax').fillna(0.0)
    w = equal_weights(metrics) if weights is None else weights.reindex(metrics).fillna(0.0)
    if float(w.sum()) <= 0:
        w = equal_weights(metrics)
    w = w / w.sum()
    score = Z.mul(w, axis=1).sum(axis=1)
    out = pd.DataFrame({product_col: df[product_col].values, 'weighted_sum_score': score})
    out['rank'] = out['weighted_sum_score'].rank(method='average', ascending=False)
    return out.sort_values(['rank', 'weighted_sum_score'], ascending=[True, False]).reset_index(drop=True)


def topsis_rank(
    df: pd.DataFrame,
    product_col: str,
    metric_directions: Mapping[str, str],
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    dirs = standardize_directions(metric_directions)
    metrics = list(dirs.keys())
    X = prepare_numeric_matrix(df, metrics)

    # vector normalization with directions handled when choosing ideal / anti-ideal
    M = X.to_numpy(dtype=float)
    denom = np.sqrt(np.nansum(M ** 2, axis=0))
    denom[~np.isfinite(denom) | (denom == 0)] = 1.0
    R = M / denom

    w = equal_weights(metrics) if weights is None else weights.reindex(metrics).fillna(0.0)
    if float(w.sum()) <= 0:
        w = equal_weights(metrics)
    w = w / w.sum()
    V = R * w.to_numpy(dtype=float)

    ideal = np.zeros(V.shape[1], dtype=float)
    nadir = np.zeros(V.shape[1], dtype=float)
    for j, m in enumerate(metrics):
        col = V[:, j]
        if dirs[m] == 'benefit':
            ideal[j] = np.nanmax(col)
            nadir[j] = np.nanmin(col)
        else:
            ideal[j] = np.nanmin(col)
            nadir[j] = np.nanmax(col)

    d_plus = np.sqrt(np.nansum((V - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.nansum((V - nadir) ** 2, axis=1))
    closeness = d_minus / (d_plus + d_minus)

    out = pd.DataFrame({product_col: df[product_col].values, 'topsis_score': closeness})
    out['rank'] = out['topsis_score'].rank(method='average', ascending=False)
    return out.sort_values(['rank', 'topsis_score'], ascending=[True, False]).reset_index(drop=True)


def consensus_rank(
    method_outputs: Mapping[str, pd.DataFrame],
    product_col: str,
    *,
    score_priority: bool = True,
) -> pd.DataFrame:
    """
    Combine multiple ranking outputs.

    Each method output must contain product_col and rank.
    Score columns, if present, are normalized to [0,1] and averaged.
    Final ranking uses average rank first; if score_priority=True, ties use consensus score.
    """
    merged = None
    for method_name, dfm in method_outputs.items():
        tmp = dfm.copy()
        if product_col not in tmp.columns or 'rank' not in tmp.columns:
            raise KeyError(f'{method_name} output must contain {product_col} and rank columns')
        score_cols = [c for c in tmp.columns if c not in {product_col, 'rank'}]
        keep = [product_col, 'rank'] + score_cols[:1]
        tmp = tmp[keep].rename(columns={'rank': f'{method_name}_rank'})
        if score_cols:
            tmp = tmp.rename(columns={score_cols[0]: f'{method_name}_score'})
        merged = tmp if merged is None else merged.merge(tmp, on=product_col, how='outer')

    rank_cols = [c for c in merged.columns if c.endswith('_rank')]
    score_cols = [c for c in merged.columns if c.endswith('_score')]
    merged['consensus_rank_mean'] = merged[rank_cols].mean(axis=1)

    if score_cols:
        norm_scores = []
        for c in score_cols:
            vals = merged[c].astype(float)
            vmin = vals.min(skipna=True)
            vmax = vals.max(skipna=True)
            if pd.isna(vmin) or pd.isna(vmax) or np.isclose(vmin, vmax):
                norm_scores.append(pd.Series(np.ones(len(vals)), index=vals.index))
            else:
                norm_scores.append((vals - vmin) / (vmax - vmin))
        merged['consensus_score'] = pd.concat(norm_scores, axis=1).mean(axis=1)
    else:
        merged['consensus_score'] = np.nan

    if score_priority:
        merged['rank'] = merged[['consensus_rank_mean', 'consensus_score']].apply(
            lambda r: r['consensus_rank_mean'], axis=1
        )
        merged = merged.sort_values(['consensus_rank_mean', 'consensus_score', product_col], ascending=[True, False, True])
        merged['rank'] = np.arange(1, len(merged) + 1)
    else:
        merged['rank'] = merged['consensus_rank_mean'].rank(method='average', ascending=True)
        merged = merged.sort_values(['rank', product_col]).reset_index(drop=True)
    return merged.reset_index(drop=True)
