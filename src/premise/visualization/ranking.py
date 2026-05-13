from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ranking_bar(df: pd.DataFrame, *, product_col: str = "product", score_col: str = "score", top_n: Optional[int] = None, ascending: bool = False, title: str = "Ranking result", figsize: Tuple[float, float] = (8, 4), out_path: Optional[str | Path] = None):
    data = df.copy()
    data = data.sort_values(score_col, ascending=ascending)
    if top_n is not None:
        data = data.head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(data[product_col].astype(str), data[score_col].values)
    ax.set_title(title)
    ax.set_ylabel(score_col)
    ax.set_xticklabels(data[product_col].astype(str), rotation=45, ha="right")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    return fig, ax


def plot_ranking_score_heatmap(df: pd.DataFrame, *, product_col: str = "product", method_col: str = "method", score_col: str = "score", title: str = "Ranking score heatmap", cmap: str = "viridis", figsize: Tuple[float, float] = (8, 5), out_path: Optional[str | Path] = None):
    pivot = df.pivot(index=product_col, columns=method_col, values=score_col)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(pivot.shape[1])); ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0])); ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    return fig, ax, pivot
