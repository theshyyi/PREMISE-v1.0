from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def _series_from_obj(obj, value_col: Optional[str] = None, time_col: Optional[str] = None):
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, xr.DataArray):
        if "time" not in obj.dims:
            raise KeyError("xarray.DataArray must contain time dimension.")
        return obj.to_series()
    if isinstance(obj, pd.DataFrame):
        if value_col is None or time_col is None:
            raise ValueError("For DataFrame input, time_col and value_col are required.")
        s = obj[[time_col, value_col]].dropna().copy()
        s[time_col] = pd.to_datetime(s[time_col])
        return s.set_index(time_col)[value_col]
    raise TypeError(f"Unsupported series type: {type(obj)!r}")


def plot_timeseries_lines(
    series_dict: Dict[str, object],
    *,
    title: str = "",
    ylabel: str = "",
    figsize: Tuple[float, float] = (8, 4),
    out_path: Optional[str | Path] = None,
):
    fig, ax = plt.subplots(figsize=figsize)
    for name, obj in series_dict.items():
        s = _series_from_obj(obj)
        ax.plot(s.index, s.values, label=name, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    return fig, ax
