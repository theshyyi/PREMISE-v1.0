from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_boxplot_groups(data_dict, *, figsize: Tuple[float, float] = (6, 4), title: Optional[str] = None, ylabel: Optional[str] = None, show_means: bool = False, out_path: Optional[str | Path] = None):
    labels = list(data_dict.keys())
    data = [np.asarray(v, dtype=float).ravel() for v in data_dict.values()]
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True, medianprops=dict(color="black"))
    for patch in bp["boxes"]:
        patch.set_facecolor("#a6cee3")
    if show_means:
        means = [np.nanmean(v) for v in data]
        ax.scatter(np.arange(1, len(labels) + 1), means, marker="D", color="red", zorder=3, label="Mean")
        ax.legend(loc="best")
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    return fig, ax


def plot_violin_groups(data_dict, *, figsize: Tuple[float, float] = (6, 4), title: Optional[str] = None, ylabel: Optional[str] = None, show_means: bool = False, out_path: Optional[str | Path] = None):
    labels = list(data_dict.keys())
    data = [np.asarray(v, dtype=float).ravel() for v in data_dict.values()]
    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#1f78b4")
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
    if show_means:
        means = [np.nanmean(v) for v in data]
        ax.scatter(np.arange(1, len(labels) + 1), means, marker="D", color="red", zorder=3, label="Mean")
        ax.legend(loc="best")
    ax.set_xticks(np.arange(1, len(labels) + 1)); ax.set_xticklabels(labels)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    return fig, ax


def plot_grouped_bar(df, *, category_col: str, value_col: str, series_col: Optional[str] = None, title: str = "", ylabel: str = "", figsize: Tuple[float, float] = (8, 4), out_path: Optional[str | Path] = None):
    import pandas as pd
    fig, ax = plt.subplots(figsize=figsize)
    if series_col is None:
        ax.bar(df[category_col].astype(str), df[value_col].values)
    else:
        pivot = df.pivot(index=category_col, columns=series_col, values=value_col)
        x = np.arange(len(pivot.index))
        n = len(pivot.columns)
        width = 0.8 / max(n, 1)
        for i, col in enumerate(pivot.columns):
            ax.bar(x + (i - (n-1)/2) * width, pivot[col].values, width=width, label=str(col))
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index.astype(str), rotation=0)
        ax.legend(frameon=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    return fig, ax


def plot_metric_heatmap(df, *, index_col: str, columns_col: str, value_col: str, title: str = "", cmap: str = "viridis", figsize: Tuple[float, float] = (7, 5), fmt: str = ".2f", annotate: bool = True, out_path: Optional[str | Path] = None):
    pivot = df.pivot(index=index_col, columns=columns_col, values=value_col)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(pivot.shape[1])); ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0])); ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(value_col, rotation=270, labelpad=12)
    if annotate:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                txt = "" if not np.isfinite(v) else format(v, fmt)
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    return fig, ax, pivot
