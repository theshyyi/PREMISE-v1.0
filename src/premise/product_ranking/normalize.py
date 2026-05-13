from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


BENEFIT_ALIASES = {'benefit', 'max', 'higher', 'higher_is_better', 'desc'}
COST_ALIASES = {'cost', 'min', 'lower', 'lower_is_better', 'asc'}


def standardize_directions(metric_directions: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in metric_directions.items():
        s = str(v).strip().lower()
        if s in BENEFIT_ALIASES:
            out[k] = 'benefit'
        elif s in COST_ALIASES:
            out[k] = 'cost'
        else:
            raise ValueError(f'Unsupported direction for {k}: {v}')
    return out


def prepare_numeric_matrix(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = df[metrics].apply(pd.to_numeric, errors='coerce')
    if out.isnull().all(axis=None):
        raise ValueError('All metric values are NaN after numeric conversion.')
    return out


def normalize_to_benefit_scale(
    metric_df: pd.DataFrame,
    metric_directions: Mapping[str, str],
    *,
    method: str = 'minmax',
) -> pd.DataFrame:
    dirs = standardize_directions(metric_directions)
    X = metric_df.copy().astype(float)
    Z = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)

    for c in X.columns:
        col = X[c].to_numpy(dtype=float)
        mask = np.isfinite(col)
        if not mask.any():
            Z[c] = np.nan
            continue

        vals = col[mask]
        cmin = np.min(vals)
        cmax = np.max(vals)
        if np.isclose(cmax, cmin):
            z = np.full_like(col, 1.0, dtype=float)
        else:
            if method == 'minmax':
                if dirs[c] == 'benefit':
                    z = (col - cmin) / (cmax - cmin)
                else:
                    z = (cmax - col) / (cmax - cmin)
            elif method == 'vector':
                denom = np.sqrt(np.nansum(col ** 2))
                if denom == 0 or not np.isfinite(denom):
                    z = np.full_like(col, np.nan, dtype=float)
                else:
                    vv = col / denom
                    if dirs[c] == 'benefit':
                        z = vv
                    else:
                        z = np.nanmax(vv) - vv
            else:
                raise ValueError(f'Unsupported normalization method: {method}')
        Z[c] = z
    return Z
