# -*- coding: utf-8 -*-

"""
plotting
========

Plotting utilities for PREMISE:

- Global font setup
- Spatial maps with shapefile overlays
- Station scatter overlays
- Taylor diagrams
- SAL diagrams
- Boxplots and violin plots for grouped data
"""

from __future__ import annotations
from typing import Optional, Sequence, Union, Dict, List, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from .preprocess import load_shapefile
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde
from matplotlib import colors
import geopandas as gpd
import regionmask



ArrayLike = Union[np.ndarray, xr.DataArray]
PathLike = Union[str, Path]

__all__ = [
    "scatter_density_product",
    "multi_scatter_density_products",
    "time_group_scatter_density_product",
    "scatter_density_product_by_regions",
    "plot_spatial_field",
    "plot_taylor_diagram",
    "plot_sal_scatter",
    "plot_boxplot_groups",
    "plot_violin_groups",
    "draw_performance_background",
    "plot_performance_diagram_seasons",
    "plot_performance_diagram_months",
    "plot_performance_diagram_elevation",
    "plot_performance_diagram_regions",
    "save_performance_diagrams_regions_by_group",
]


# ========== Fonts and styles ==========

def setup_mpl_fonts(
    use_chinese: bool = True,
    base_size: int = 10,
) -> None:
    """
    Set global Matplotlib fonts.

    - English: Times New Roman
    - Optional Chinese: SimHei
    """
    from matplotlib import rcParams

    rcParams["font.family"] = "sans-serif"
    if use_chinese:
        rcParams["font.sans-serif"] = ["Times New Roman", "SimHei"]
    else:
        rcParams["font.sans-serif"] = ["Times New Roman"]

    rcParams["axes.titlesize"] = base_size + 1
    rcParams["axes.labelsize"] = base_size
    rcParams["xtick.labelsize"] = base_size - 1
    rcParams["ytick.labelsize"] = base_size - 1
    rcParams["legend.fontsize"] = base_size - 1


# ========== Spatial map plotting ==========

def plot_spatial_field(
    da: xr.DataArray,
    *,
    lon_name: str = "lon",
    lat_name: str = "lat",
    shp_path: Optional[str] = None,
    region_field: Optional[str] = None,
    region_values: Optional[Sequence[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    cbar_label: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
    coastlines: bool = True,
    gridlines: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot a 2D spatial field with optional shapefile overlays.
    """
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": proj},
    )

    lon = da[lon_name].values
    lat = da[lat_name].values

    data_2d = da
    if da.ndim > 2:
        data_2d = da.isel({da.dims[0]: 0})

    lon2d, lat2d = np.meshgrid(lon, lat)

    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        data_2d.values,
        transform=proj,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if coastlines:
        ax.coastlines(resolution="50m", linewidth=0.5)

    if gridlines:
        gl = ax.gridlines(
            draw_labels=True,
            x_inline=False,
            y_inline=False,
            linewidth=0.3,
            alpha=0.5,
        )
        gl.top_labels = False
        gl.right_labels = False

    if shp_path is not None:
        gdf = load_shapefile(shp_path)

        if region_field is not None and region_values is not None:
            if isinstance(region_values, str):
                region_values = [region_values]
            gdf = gdf[gdf[region_field].isin(region_values)]

        if not gdf.empty:
            feat = ShapelyFeature(
                gdf.geometry,
                crs=proj,
                edgecolor="black",
                facecolor="none",
                linewidth=0.8,
            )
            ax.add_feature(feat)

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="vertical",
        fraction=0.045,
        pad=0.03,
    )
    if cbar_label is not None:
        cbar.set_label(cbar_label)

    if title is not None:
        ax.set_title(title)

    return fig, ax


def add_station_scatter(
    ax: Axes,
    station_df,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    value_col: Optional[str] = None,
    s: float = 10.0,
    marker: str = "o",
    edgecolor: str = "k",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    label: Optional[str] = None,
    transform: ccrs.CRS = ccrs.PlateCarree(),
):
    """
    Overlay station locations on an existing Cartopy map.
    """
    lats = station_df[lat_col].values
    lons = station_df[lon_col].values

    if value_col is None:
        sc = ax.scatter(
            lons,
            lats,
            s=s,
            marker=marker,
            edgecolor=edgecolor,
            facecolor="none",
            transform=transform,
            label=label,
        )
        return sc, None
    else:
        values = station_df[value_col].values
        sc = ax.scatter(
            lons,
            lats,
            c=values,
            s=s,
            marker=marker,
            edgecolor=edgecolor,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=transform,
            label=label,
        )
        cbar = plt.colorbar(sc, ax=ax, orientation="vertical", fraction=0.045, pad=0.03)
        return sc, cbar


# ========== Taylor diagram ==========

import math


def plot_taylor_diagram(
    ref_std: float,
    stds,
    corrs,
    labels,
    *,
    fig: Optional[Figure] = None,
    rect: int | str = 111,
    title: Optional[str] = None,
    marker: str = "o",
):
    """
    Draw a simple Taylor diagram in polar coordinates.
    """
    stds = np.asarray(stds, dtype=float)
    corrs = np.clip(np.asarray(corrs, dtype=float), -1.0, 1.0)
    labels = list(labels)

    if fig is None:
        fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(rect, polar=True)

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("E")

    max_std = max(ref_std, np.nanmax(stds)) * 1.1

    ax.plot(0, ref_std, "ko", label="REF")

    for s, c, lab in zip(stds, corrs, labels):
        if not np.isfinite(s) or not np.isfinite(c):
            continue
        theta = math.acos(c)
        ax.plot(theta, s, marker, label=lab)

    corr_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    theta_ticks = [math.degrees(math.acos(c)) for c in corr_levels]
    ax.set_thetagrids(theta_ticks, [f"{c:.1f}" for c in corr_levels])

    r_levels = np.linspace(0, max_std, 4)[1:]
    ax.set_rlim(0, max_std)
    ax.set_rgrids(
        r_levels,
        labels=[f"{v:.2f}" for v in r_levels],
    )
    ax.set_ylabel("Standard deviation", labelpad=20)

    if title is not None:
        ax.set_title(title, pad=20)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig, ax


# ========== SAL diagram ==========

def plot_sal_scatter(
    S,
    A,
    L=None,
    labels=None,
    *,
    figsize: Tuple[float, float] = (6, 6),
    title: Optional[str] = "SAL diagram",
    cmap: str = "viridis",
):
    """
    Draw a SAL (Structure–Amplitude–Location) diagram.
    """
    S = np.asarray(S, dtype=float)
    A = np.asarray(A, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)

    if L is None:
        sc = ax.scatter(A, S, s=40, edgecolor="k", facecolor="none")
        cbar = None
    else:
        L = np.asarray(L, dtype=float)
        sc = ax.scatter(A, S, c=L, s=40, edgecolor="k", cmap=cmap)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("L")

    if labels is not None:
        for x, y, lab in zip(A, S, labels):
            ax.text(x, y, lab, fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("Amplitude (A)")
    ax.set_ylabel("Structure (S)")
    ax.set_title(title)

    ax.set_xlim(min(-2, np.nanmin(A) - 0.5), max(2, np.nanmax(A) + 0.5))
    ax.set_ylim(min(-2, np.nanmin(S) - 0.5), max(2, np.nanmax(S) + 0.5))

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    return fig, ax


# ========== Boxplot / Violin plot ==========
def plot_boxplot_groups(
    data_dict,
    *,
    figsize: Tuple[float, float] = (6, 4),
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_means: bool = False,
):
    """
    Boxplots for multiple groups.
    """
    labels = list(data_dict.keys())
    data = [np.asarray(v, dtype=float).ravel() for v in data_dict.values()]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black"),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("#a6cee3")

    if show_means:
        means = [np.nanmean(v) for v in data]
        ax.scatter(
            np.arange(1, len(labels) + 1),
            means,
            marker="D",
            color="red",
            zorder=3,
            label="Mean",
        )
        ax.legend(loc="best")

    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    return fig, ax


def plot_violin_groups(
    data_dict,
    *,
    figsize: Tuple[float, float] = (6, 4),
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_means: bool = False,
):
    """
    Violin plots for multiple groups.
    """
    labels = list(data_dict.keys())
    data = [np.asarray(v, dtype=float).ravel() for v in data_dict.values()]

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for pc in parts["bodies"]:
        pc.set_facecolor("#1f78b4")
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)

    if "cmedians" in parts:
        parts["cmedians"].set_color("black")

    if show_means:
        means = [np.nanmean(v) for v in data]
        ax.scatter(
            np.arange(1, len(labels) + 1),
            means,
            marker="D",
            color="red",
            zorder=3,
            label="Mean",
        )
        ax.legend(loc="best")

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0)

    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    return fig, ax


def _flatten_valid(ref: ArrayLike, test: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """拉平两个数组并去掉 NaN / inf，返回 1D x,y。"""
    if isinstance(ref, xr.DataArray):
        x = ref.values.ravel()
    else:
        x = np.asarray(ref).ravel()

    if isinstance(test, xr.DataArray):
        y = test.values.ravel()
    else:
        y = np.asarray(test).ravel()

    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _density_and_stats(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """给定 1D x,y，返回点密度 z 和统计量 dict。"""
    if x.size == 0:
        raise ValueError("density_and_stats: 有效数据为空。")

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)  # 点密度

    diff = y - x
    bias = float(diff.mean())
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    r = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else np.nan

    denom = float(np.sum(x ** 2))
    slope = float(np.sum(x * y) / denom) if denom > 0 else np.nan

    stats = {
        "N": int(x.size),
        "bias": bias,
        "rmse": rmse,
        "r": r,
        "slope": slope,
    }
    return z, stats


def scatter_density_product(
    ref: ArrayLike,
    test: ArrayLike,
    *,
    x_label: str,
    y_label: str,
    title: str = "",
    out_path: PathLike | None = None,
    cmap: str = "gist_rainbow",
    point_size: float = 5.0,
    tick_step: float = 5.0,
    figsize: Tuple[float, float] = (8.0, 6.0),
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, float]]:
    """
    单个产品 vs 参考的密度散点图。

    新增参数
    --------
    figsize : (float, float), default (8, 6)
        图像尺寸（inch）。
    dpi : int, default 600
        分辨率。
    """
    x, y = _flatten_valid(ref, test)
    z, stats = _density_and_stats(x, y)

    max_val = max(x.max(), y.max())
    min_val = min(x.min(), y.min(), 0.0)
    upper = np.ceil(max_val / tick_step) * tick_step
    if upper <= 0:
        upper = tick_step

    x_line = np.linspace(min_val, upper, 200)
    y_line = stats["slope"] * x_line

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    norm = colors.Normalize(vmin=0.0, vmax=z.max())
    sc = ax.scatter(
        x,
        y,
        c=z * 100.0,
        s=point_size,
        marker="o",
        edgecolors="none",
        cmap=cmap,
        norm=norm,
        label="Grid point",
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        orientation="vertical",
        shrink=1.0,
        pad=0.015,
        aspect=30,
        extend="both",
    )
    cbar.set_label("Point density")

    ax.plot(x_line, y_line, color="red", lw=1.5, label=f"Fit: y = {stats['slope']:.3f}x")
    ax.plot([min_val, upper], [min_val, upper], "k--", lw=1.5, label="1:1 line")

    ax.set_xlim(min_val, upper)
    ax.set_ylim(min_val, upper)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.tick_params(which="major", width=2.5, length=5)

    for spine in ["bottom", "top", "left", "right"]:
        ax.spines[spine].set_linewidth(2.5)

    dx = upper - min_val
    text_x = min_val + 0.05 * dx
    text_y_top = min_val + 0.95 * dx

    ax.text(text_x, text_y_top, rf"$N={stats['N']}$")
    ax.text(text_x + 0.35 * dx, text_y_top, rf"$R={stats['r']:.2f}$")
    ax.text(text_x, text_y_top - 0.05 * dx, rf"$BIAS={stats['bias']:.2f}$")
    ax.text(text_x, text_y_top - 0.10 * dx, rf"$RMSE={stats['rmse']:.2f}$")

    if title:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    return fig, ax, stats

def multi_scatter_density_products(
    ref: ArrayLike,
    tests: Dict[str, ArrayLike],
    *,
    x_label: str,
    y_label: str,
    title: str = "",
    out_path: PathLike | None = None,
    cmap: str = "gist_rainbow",
    point_size: float = 4.0,
    tick_step: float = 5.0,
    ncols: int | None = None,
    panel_size: Tuple[float, float] = (4.0, 4.0),
    figsize: Tuple[float, float] | None = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Dict[str, float]]]:
    """
    多产品 vs 同一参考产品的密度散点图，多子图布局。

    新增参数
    --------
    panel_size : (float, float), default (4, 4)
        当 figsize 未指定时，每个子图的基准尺寸（inch）。
    figsize : (float, float), optional
        整张图像尺寸。如果给定，则忽略 panel_size×(ncols,nrows) 的自动计算。
    dpi : int, default 600
        分辨率。
    """
    labels = list(tests.keys())
    n_prod = len(labels)
    if n_prod == 0:
        raise ValueError("tests 为空。")

    if ncols is None:
        if n_prod <= 2:
            ncols = n_prod
        elif n_prod == 3:
            ncols = 3
        else:
            ncols = 2
    nrows = int(np.ceil(n_prod / ncols))

    # 预计算 xy/z/stats
    xy_all: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    z_all: Dict[str, np.ndarray] = {}
    stats_all: Dict[str, Dict[str, float]] = {}
    x_min_list = []
    x_max_list = []
    y_min_list = []
    y_max_list = []
    z_max_list = []

    for name in labels:
        x, y = _flatten_valid(ref, tests[name])
        if x.size == 0:
            raise ValueError(f"产品 {name} 有效数据为空。")
        z, stats = _density_and_stats(x, y)

        xy_all[name] = (x, y)
        z_all[name] = z
        stats_all[name] = stats

        x_min_list.append(x.min())
        x_max_list.append(x.max())
        y_min_list.append(y.min())
        y_max_list.append(y.max())
        z_max_list.append(z.max())

    max_val = max(max(x_max_list), max(y_max_list))
    min_val = min(min(x_min_list), min(y_min_list), 0.0)
    upper = np.ceil(max_val / tick_step) * tick_step
    if upper <= 0:
        upper = tick_step

    x_line = np.linspace(min_val, upper, 200)
    z_global_max = max(z_max_list)
    norm = colors.Normalize(vmin=0.0, vmax=z_global_max)

    if figsize is None:
        fig_w = panel_size[0] * ncols
        fig_h = panel_size[1] * nrows
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )

    first_sc = None
    for idx, name in enumerate(labels):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        x, y = xy_all[name]
        z = z_all[name]
        stats = stats_all[name]

        sc = ax.scatter(
            x,
            y,
            c=z * 100.0,
            s=point_size,
            marker="o",
            edgecolors="none",
            cmap=cmap,
            norm=norm,
            label="Grid point",
        )
        if first_sc is None:
            first_sc = sc

        y_line = stats["slope"] * x_line
        ax.plot(x_line, y_line, color="red", lw=1.5, label=f"Fit: y = {stats['slope']:.3f}x")
        ax.plot([min_val, upper], [min_val, upper], "k--", lw=1.5, label="1:1 line")

        ax.set_xlim(min_val, upper)
        ax.set_ylim(min_val, upper)
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        ax.tick_params(which="major", width=2.0, length=4)

        for spine in ["bottom", "top", "left", "right"]:
            ax.spines[spine].set_linewidth(2.0)

        dx = upper - min_val
        text_x = min_val + 0.05 * dx
        text_y_top = min_val + 0.95 * dx

        ax.text(text_x, text_y_top, rf"$N={stats['N']}$", fontsize=10)
        ax.text(text_x + 0.35 * dx, text_y_top, rf"$R={stats['r']:.2f}$", fontsize=10)
        ax.text(text_x, text_y_top - 0.05 * dx, rf"$BIAS={stats['bias']:.2f}$", fontsize=10)
        ax.text(text_x, text_y_top - 0.10 * dx, rf"$RMSE={stats['rmse']:.2f}$", fontsize=10)

        ax.set_title(name, fontsize=12)

        if row == nrows - 1:
            ax.set_xlabel(x_label)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(y_label)
        else:
            ax.set_yticklabels([])

        if idx == 0:
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.legend().set_visible(False)

    # 隐藏空子图
    for j in range(n_prod, nrows * ncols):
        row = j // ncols
        col = j % ncols
        axes[row, col].axis("off")

    # 共用 colorbar
    if first_sc is not None:
        cbar = fig.colorbar(
            first_sc,
            ax=axes.ravel().tolist(),
            orientation="vertical",
            shrink=0.9,
            pad=0.03,
            aspect=30,
            extend="both",
        )
        cbar.set_label("Point density")

    if title:
        fig.suptitle(title, y=0.98)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    return fig, axes, stats_all

def time_group_scatter_density_product(
    ref: xr.DataArray,
    test: xr.DataArray,
    *,
    group: str = "season",
    time_dim: str = "time",
    groups=None,
    group_labels=None,
    x_label: str = "Reference climatology (mm)",
    y_label: str = "Test climatology (mm)",
    title: str = "",
    out_path: PathLike | None = None,
    cmap: str = "gist_rainbow",
    point_size: float = 4.0,
    tick_step: float = 5.0,
    ncols: int | None = None,
    panel_size: Tuple[float, float] = (4.0, 4.0),
    figsize: Tuple[float, float] | None = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Dict[str, float]]]:
    """
    同一对产品，按时间分组（季节 / 月份）分别算 climatology，多面板散点图。

    新增参数
    --------
    panel_size : (float, float), default (4, 4)
        未显式提供 figsize 时，每个子图的基准尺寸。
    figsize : (float, float), optional
        整张图像的尺寸（inch）。若提供，则覆盖 panel_size×(ncols,nrows)。
    dpi : int, default 600
        分辨率。
    """
    ref, test = xr.align(ref, test, join="inner")
    time = ref[time_dim]

    if group == "season":
        coord = time.dt.season
        default_order = ["DJF", "MAM", "JJA", "SON"]
    elif group == "month":
        coord = time.dt.month
        default_order = list(range(1, 13))
    else:
        raise ValueError(f"Unsupported group='{group}', only 'season' or 'month' allowed.")

    actual_values = list(np.unique(coord.values))
    if groups is None:
        groups = [g for g in default_order if g in actual_values] or actual_values

    if group_labels is None:
        if group == "season":
            group_labels = groups
        else:
            month_name = {
                1: "Jan",  2: "Feb",  3: "Mar",  4: "Apr",
                5: "May",  6: "Jun",  7: "Jul",  8: "Aug",
                9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
            }
            group_labels = [month_name.get(m, str(m)) for m in groups]

    if len(groups) != len(group_labels):
        raise ValueError("groups 与 group_labels 长度不一致。")

    clim_x: Dict[str, xr.DataArray] = {}
    clim_y: Dict[str, xr.DataArray] = {}
    label_by_group = dict(zip(groups, group_labels))

    for g in groups:
        mask = coord == g
        if mask.sum() == 0:
            continue
        ref_g = ref.sel({time_dim: mask}).mean(time_dim, skipna=True)
        test_g = test.sel({time_dim: mask}).mean(time_dim, skipna=True)
        lab = label_by_group[g]
        clim_x[lab] = ref_g
        clim_y[lab] = test_g

    if not clim_x:
        raise ValueError("time_group_scatter_density_product: 所有分组均无有效数据。")

    xy_all: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    z_all: Dict[str, np.ndarray] = {}
    stats_all: Dict[str, Dict[str, float]] = {}
    x_min_list = []
    x_max_list = []
    y_min_list = []
    y_max_list = []
    z_max_list = []

    for lab in clim_x.keys():
        x, y = _flatten_valid(clim_x[lab], clim_y[lab])
        if x.size == 0:
            continue
        z, stats = _density_and_stats(x, y)
        xy_all[lab] = (x, y)
        z_all[lab] = z
        stats_all[lab] = stats

        x_min_list.append(x.min())
        x_max_list.append(x.max())
        y_min_list.append(y.min())
        y_max_list.append(y.max())
        z_max_list.append(z.max())

    if not xy_all:
        raise ValueError("time_group_scatter_density_product: 展平后所有分组均无有效数据。")

    max_val = max(max(x_max_list), max(y_max_list))
    min_val = min(min(x_min_list), min(y_min_list), 0.0)
    upper = np.ceil(max_val / tick_step) * tick_step
    if upper <= 0:
        upper = tick_step

    x_line = np.linspace(min_val, upper, 200)
    z_global_max = max(z_max_list)
    norm = colors.Normalize(vmin=0.0, vmax=z_global_max)

    labels = list(xy_all.keys())
    n_panels = len(labels)
    if ncols is None:
        if n_panels <= 2:
            ncols = n_panels
        elif n_panels <= 4:
            ncols = 2
        elif n_panels <= 6:
            ncols = 3
        else:
            ncols = 4
    nrows = int(np.ceil(n_panels / ncols))

    if figsize is None:
        fig_w = panel_size[0] * ncols
        fig_h = panel_size[1] * nrows
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    first_sc = None
    for idx, lab in enumerate(labels):
        ax = axes_flat[idx]
        x, y = xy_all[lab]
        z = z_all[lab]
        stats = stats_all[lab]

        sc = ax.scatter(
            x,
            y,
            c=z * 100.0,
            s=point_size,
            marker="o",
            edgecolors="none",
            cmap=cmap,
            norm=norm,
            label="Grid point",
        )
        if first_sc is None:
            first_sc = sc

        y_line = stats["slope"] * x_line
        ax.plot(x_line, y_line, color="red", lw=1.5, label=f"Fit: y = {stats['slope']:.3f}x")
        ax.plot([min_val, upper], [min_val, upper], "k--", lw=1.5, label="1:1 line")

        ax.set_xlim(min_val, upper)
        ax.set_ylim(min_val, upper)
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        ax.tick_params(which="major", width=2.0, length=4)

        for spine in ["bottom", "top", "left", "right"]:
            ax.spines[spine].set_linewidth(2.0)

        dx = upper - min_val
        text_x = min_val + 0.05 * dx
        text_y_top = min_val + 0.95 * dx

        ax.text(text_x, text_y_top, rf"$N={stats['N']}$", fontsize=10)
        ax.text(text_x + 0.35 * dx, text_y_top, rf"$R={stats['r']:.2f}$", fontsize=10)
        ax.text(text_x, text_y_top - 0.05 * dx, rf"$BIAS={stats['bias']:.2f}$", fontsize=10)
        ax.text(text_x, text_y_top - 0.10 * dx, rf"$RMSE={stats['rmse']:.2f}$", fontsize=10)

        ax.set_title(lab, fontsize=12)

        row = idx // ncols
        col = idx % ncols
        if row == nrows - 1:
            ax.set_xlabel(x_label)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(y_label)
        else:
            ax.set_yticklabels([])

        if idx == 0:
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.legend().set_visible(False)

    # 隐藏多余子图
    for j in range(n_panels, nrows * ncols):
        axes_flat[j].axis("off")

    if first_sc is not None:
        cbar = fig.colorbar(
            first_sc,
            ax=axes_flat.tolist(),
            orientation="vertical",
            shrink=0.9,
            pad=0.03,
            aspect=30,
            extend="both",
        )
        cbar.set_label("Point density")

    if title:
        fig.suptitle(title, y=0.98)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    return fig, axes, stats_all


def scatter_density_product_by_regions(
    ref_clim: xr.DataArray,
    test_clim: xr.DataArray,
    *,
    shp_path: PathLike,
    region_field: str,
    x_label: str = "Reference climatology (mm)",
    y_label: str = "Test climatology (mm)",
    title: str = "",
    out_path: PathLike | None = None,
    cmap: str = "gist_rainbow",
    point_size: float = 3.0,
    tick_step: float = 5.0,
    ncols: int = 3,
    panel_size: Tuple[float, float] = (4.0, 4.0),
    figsize: Tuple[float, float] | None = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Dict[str, float]]]:
    """
    同一对产品的 climatology，按 shapefile 定义的区域分区分别绘制散点，多面板图。

    新增参数
    --------
    panel_size : (float, float), default (4, 4)
        未显式提供 figsize 时，每个子图的基准尺寸。
    figsize : (float, float), optional
        整张图像尺寸（inch）。若指定，则覆盖 panel_size×(ncols,nrows)。
    dpi : int, default 600
        分辨率。
    """
    ref_clim, test_clim = xr.align(ref_clim, test_clim, join="inner")

    shp_path = Path(shp_path)
    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    if region_field not in gdf.columns:
        raise KeyError(f"shapefile 中未找到字段 '{region_field}'。")

    region_names = gdf[region_field].astype(str).tolist()
    regions = regionmask.Regions(
        list(gdf.geometry.values),
        names=region_names,
        numbers=list(range(len(region_names))),
    )

    lon = ref_clim["lon"].values
    lat = ref_clim["lat"].values
    mask = regions.mask(lon, lat)

    xy_all: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    z_all: Dict[str, np.ndarray] = {}
    stats_all: Dict[str, Dict[str, float]] = {}

    x_min_list = []
    x_max_list = []
    y_min_list = []
    y_max_list = []
    z_max_list = []

    for i, name in enumerate(region_names):
        region_mask = (mask == i)
        ref_reg = ref_clim.where(region_mask)
        test_reg = test_clim.where(region_mask)

        x, y = _flatten_valid(ref_reg, test_reg)
        if x.size == 0:
            continue

        z, stats = _density_and_stats(x, y)
        xy_all[name] = (x, y)
        z_all[name] = z
        stats_all[name] = stats

        x_min_list.append(x.min())
        x_max_list.append(x.max())
        y_min_list.append(y.min())
        y_max_list.append(y.max())
        z_max_list.append(z.max())

    if not xy_all:
        raise ValueError("scatter_density_product_by_regions: 所有区域均无有效数据。")

    max_val = max(max(x_max_list), max(y_max_list))
    min_val = min(min(x_min_list), min(y_min_list), 0.0)
    upper = np.ceil(max_val / tick_step) * tick_step
    if upper <= 0:
        upper = tick_step

    x_line = np.linspace(min_val, upper, 200)
    z_global_max = max(z_max_list)
    norm = colors.Normalize(vmin=0.0, vmax=z_global_max)

    labels = list(xy_all.keys())
    n_regions = len(labels)
    nrows = int(np.ceil(n_regions / ncols))

    if figsize is None:
        fig_w = panel_size[0] * ncols
        fig_h = panel_size[1] * nrows
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    first_sc = None

    for idx, name in enumerate(labels):
        ax = axes_flat[idx]
        x, y = xy_all[name]
        z = z_all[name]
        stats = stats_all[name]

        sc = ax.scatter(
            x,
            y,
            c=z * 100.0,
            s=point_size,
            marker="o",
            edgecolors="none",
            cmap=cmap,
            norm=norm,
            label="Grid point",
        )
        if first_sc is None:
            first_sc = sc

        y_line = stats["slope"] * x_line
        ax.plot(x_line, y_line, color="red", lw=1.3, label=f"Fit: y = {stats['slope']:.3f}x")
        ax.plot([min_val, upper], [min_val, upper], "k--", lw=1.3, label="1:1 line")

        ax.set_xlim(min_val, upper)
        ax.set_ylim(min_val, upper)
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        ax.tick_params(which="major", width=1.8, length=3.5)

        for spine in ["bottom", "top", "left", "right"]:
            ax.spines[spine].set_linewidth(1.8)

        dx = upper - min_val
        text_x = min_val + 0.05 * dx
        text_y_top = min_val + 0.95 * dx

        ax.text(text_x, text_y_top, rf"$N={stats['N']}$", fontsize=8)
        ax.text(text_x + 0.35 * dx, text_y_top, rf"$R={stats['r']:.2f}$", fontsize=8)
        ax.text(text_x, text_y_top - 0.05 * dx, rf"$BIAS={stats['bias']:.2f}$", fontsize=8)
        ax.text(text_x, text_y_top - 0.10 * dx, rf"$RMSE={stats['rmse']:.2f}$", fontsize=8)

        ax.set_title(name, fontsize=10)

        row = idx // ncols
        col = idx % ncols
        if row == nrows - 1:
            ax.set_xlabel(x_label, fontsize=9)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(y_label, fontsize=9)
        else:
            ax.set_yticklabels([])

        if idx == 0:
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.legend().set_visible(False)

    for j in range(n_regions, nrows * ncols):
        axes_flat[j].axis("off")

    if first_sc is not None:
        cbar = fig.colorbar(
            first_sc,
            ax=axes_flat.tolist(),
            orientation="vertical",
            shrink=0.9,
            pad=0.03,
            aspect=30,
            extend="both",
        )
        cbar.set_label("Point density")

    if title:
        fig.suptitle(title, y=0.98)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    return fig, axes, stats_all



def draw_performance_background(
    ax: plt.Axes,
    *,
    csi_min: float = 0.1,
    csi_max: float = 0.9,
    csi_step: float = 0.1,
    cmap: str = "Blues",
    bias_values: Sequence[float] = (0.5, 0.7, 1.0, 1.5, 2.0),
) -> plt.contourf:
    """
    在给定 Axes 上绘制 POD–(1-FAR) 的 CSI 背景 + CSI 等值线 + bias 线。
    返回 contourf 对象，用于统一设置 colorbar。
    """
    sr = np.linspace(0.01, 1.0, 200)
    pod = np.linspace(0.01, 1.0, 200)
    SR, POD = np.meshgrid(sr, pod)

    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    CSI[(CSI < 0) | ~np.isfinite(CSI)] = np.nan

    levels = np.linspace(csi_min, csi_max, int((csi_max - csi_min) / csi_step) + 1)
    cf = ax.contourf(
        SR,
        POD,
        CSI,
        levels=levels,
        cmap=cmap,
        extend="both",
    )

    cs = ax.contour(
        SR,
        POD,
        CSI,
        levels=levels,
        colors="k",
        linewidths=0.6,
        linestyles="dashed",
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    # bias 线
    for b in bias_values:
        sr_line = np.linspace(0.01, 1.0, 200)
        pod_line = b * sr_line
        pod_line[pod_line > 1.0] = np.nan
        ax.plot(sr_line, pod_line, "k--", linewidth=0.8)
        idx = np.nanargmax(pod_line)
        if not np.isnan(pod_line[idx]):
            ax.text(
                sr_line[idx],
                pod_line[idx] + 0.02,
                f"{b:.1f}",
                fontsize=7,
                ha="center",
                va="bottom",
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Success Ratio (1 - FAR)")
    ax.set_ylabel("Probability of Detection (POD)")
    ax.grid(False)

    return cf


def plot_performance_diagram_seasons(
    season_stats: Dict[str, Dict[str, Tuple[float, float]]],
    products: List[str],
    *,
    season_order: Sequence[str] = ("Spring", "Summer", "Autumn", "Winter"),
    panel_labels: Sequence[str] | None = None,
    figsize: Tuple[float, float] = (9.0, 8.0),
    dpi: int = 600,
    marker_size: float = 60.0,
    cmap_products: str = "tab10",
    out_path: str | Path | None = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    绘制 2×2 季节 performance diagram。

    season_stats[season][product] = (POD, FAR)
    """
    if panel_labels is None:
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.ravel()

    n_prod = len(products)
    cmap = plt.cm.get_cmap(cmap_products, n_prod)
    markers = ["o", "s", "^", "P", "X", "D", "*", "v", "<", ">"]
    color_map = {p: cmap(i) for i, p in enumerate(products)}
    marker_map = {p: markers[i % len(markers)] for i, p in enumerate(products)}

    cf_for_cbar = None

    for i, (season, lab) in enumerate(zip(season_order, panel_labels)):
        ax = axes[i]
        cf = draw_performance_background(ax)
        if cf_for_cbar is None:
            cf_for_cbar = cf

        for p in products:
            pod, far = season_stats.get(season, {}).get(p, (np.nan, np.nan))
            if np.isnan(pod) or np.isnan(far):
                continue
            sr = 1.0 - far
            ax.scatter(
                sr,
                pod,
                marker=marker_map[p],
                color=color_map[p],
                s=marker_size,
                edgecolors="k",
                linewidths=0.6,
                zorder=5,
            )

        ax.text(
            0.02,
            0.95,
            f"{lab} {season}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )
        ax.tick_params(labelsize=8)

    # legend
    handles = []
    labels = []
    for p in products:
        h = plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=marker_map[p],
            markersize=6,
            markerfacecolor=color_map[p],
            markeredgecolor="k",
            label=p,
        )
        handles.append(h)
        labels.append(p)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(products), 6),
        frameon=False,
        fontsize=7,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    # colorbar
    cax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
    cbar = fig.colorbar(cf_for_cbar, cax=cax)
    cbar.set_label("Critical Success Index (CSI)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0.03, 0.08, 0.90, 0.96])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    return fig, axes.reshape(2, 2)


def plot_performance_diagram_months(
    month_stats: Dict[int, Dict[str, Tuple[float, float]]],
    products: List[str],
    *,
    months: Sequence[int] | None = None,
    figsize: Tuple[float, float] = (10.0, 12.0),
    dpi: int = 600,
    marker_size: float = 40.0,
    cmap_products: str = "tab10",
    out_path: str | Path | None = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    绘制 4×3 月度 performance diagram。
    """
    if months is None:
        months = list(range(1, 13))
    panel_labels = [f"({chr(97+i)})" for i in range(len(months))]

    fig, axes = plt.subplots(4, 3, figsize=figsize, dpi=dpi, sharex=True, sharey=True)
    axes = axes.ravel()

    n_prod = len(products)
    cmap = plt.cm.get_cmap(cmap_products, n_prod)
    markers = ["o", "s", "^", "P", "X", "D", "*", "v", "<", ">"]
    color_map = {p: cmap(i) for i, p in enumerate(products)}
    marker_map = {p: markers[i % len(markers)] for i, p in enumerate(products)}

    cf_for_cbar = None

    for i, (m, lab) in enumerate(zip(months, panel_labels)):
        ax = axes[i]
        cf = draw_performance_background(ax)
        if cf_for_cbar is None:
            cf_for_cbar = cf

        for p in products:
            pod, far = month_stats.get(m, {}).get(p, (np.nan, np.nan))
            if np.isnan(pod) or np.isnan(far):
                continue
            sr = 1.0 - far
            ax.scatter(
                sr,
                pod,
                marker=marker_map[p],
                color=color_map[p],
                s=marker_size,
                edgecolors="k",
                linewidths=0.5,
                zorder=5,
            )

        ax.text(
            0.02,
            0.95,
            f"{lab} M{m:02d}",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
        )
        ax.tick_params(labelsize=7)

    # 去掉多余标签（保持和你原图类似的干净布局）
    for ax in axes[:-3]:
        ax.set_xlabel("")
    for ax in axes[1:]:
        ax.set_ylabel("")

    axes[-2].set_xlabel("Success Ratio (1 - FAR)", fontsize=9)
    axes[3].set_ylabel("Probability of Detection (POD)", fontsize=9)

    # legend
    handles = []
    labels = []
    for p in products:
        h = plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=marker_map[p],
            markersize=6,
            markerfacecolor=color_map[p],
            markeredgecolor="k",
            label=p,
        )
        handles.append(h)
        labels.append(p)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(products), 6),
        frameon=False,
        fontsize=9,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    cax = fig.add_axes([0.92, 0.14, 0.02, 0.70])
    cbar = fig.colorbar(cf_for_cbar, cax=cax)
    cbar.set_label("Critical Success Index (CSI)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0.03, 0.08, 0.90, 0.96])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    return fig, axes.reshape(4, 3)


def plot_performance_diagram_elevation(
    elev_labels: List[str],
    elev_stats: Dict[str, Dict[str, Tuple[float, float]]],
    products: List[str],
    *,
    ncols: int = 2,
    figsize: Tuple[float, float] | None = None,
    panel_size: Tuple[float, float] = (4.5, 4.0),
    dpi: int = 600,
    marker_size: float = 50.0,
    cmap_products: str = "tab10",
    out_path: str | Path | None = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    多面板 POD–(1-FAR) 图，每个面板对应一个海拔带。
    """
    n_band = len(elev_labels)
    nrows = int(np.ceil(n_band / ncols))

    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.array(axes).reshape(nrows, ncols)

    n_prod = len(products)
    cmap = plt.cm.get_cmap(cmap_products, n_prod)
    markers = ["o", "s", "^", "P", "X", "D", "*", "v", "<", ">"]
    color_map = {p: cmap(i) for i, p in enumerate(products)}
    marker_map = {p: markers[i % len(markers)] for i, p in enumerate(products)}

    panel_labels = [f"({chr(97+i)})" for i in range(n_band)]
    cf_for_cbar = None

    for idx, (lab, plab) in enumerate(zip(elev_labels, panel_labels)):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        cf = draw_performance_background(ax)
        if cf_for_cbar is None:
            cf_for_cbar = cf

        for p in products:
            pod, far = elev_stats.get(lab, {}).get(p, (np.nan, np.nan))
            if np.isnan(pod) or np.isnan(far):
                continue
            sr = 1.0 - far
            ax.scatter(
                sr,
                pod,
                marker=marker_map[p],
                color=color_map[p],
                s=marker_size,
                edgecolors="k",
                linewidths=0.6,
                zorder=5,
            )

        ax.text(
            0.02,
            0.95,
            f"{plab} {lab}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )

    # 删除多余子图
    if n_band < nrows * ncols:
        for k in range(n_band, nrows * ncols):
            r = k // ncols
            c = k % ncols
            fig.delaxes(axes[r, c])

    # legend
    handles = []
    labels = []
    for p in products:
        h = plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=marker_map[p],
            markersize=6,
            markerfacecolor=color_map[p],
            markeredgecolor="k",
            label=p,
        )
        handles.append(h)
        labels.append(p)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(products), 5),
        frameon=False,
        fontsize=8,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    cax = fig.add_axes([0.92, 0.20, 0.02, 0.60])
    cbar = fig.colorbar(cf_for_cbar, cax=cax)
    cbar.set_label("Critical Success Index (CSI)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0.05, 0.08, 0.90, 0.96])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    return fig, axes


def plot_performance_diagram_regions(
    region_stats: Dict[str, Dict[str, Tuple[float, float]]],
    products: List[str],
    *,
    region_order: List[str] | None = None,
    ncols: int = 3,
    figsize: Tuple[float, float] | None = None,
    panel_size: Tuple[float, float] = (4.5, 4.0),
    dpi: int = 600,
    marker_size: float = 50.0,
    cmap_products: str = "tab10",
    out_path: str | Path | None = None,
    title: str = "",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    按分区绘制 performance diagram：每个面板一个分区，点为不同产品。

    region_stats[region][product] = (POD, FAR)
    """
    if region_order is None:
        region_order = list(region_stats.keys())

    n_region = len(region_order)
    nrows = int(np.ceil(n_region / ncols))

    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.array(axes).reshape(nrows, ncols)

    n_prod = len(products)
    cmap = plt.cm.get_cmap(cmap_products, max(n_prod, 1))
    markers = ["o", "s", "^", "P", "X", "D", "*", "v", "<", ">"]
    color_map = {p: cmap(i) for i, p in enumerate(products)}
    marker_map = {p: markers[i % len(markers)] for i, p in enumerate(products)}

    panel_labels = [f"({chr(97+i)})" for i in range(n_region)]
    cf_for_cbar = None

    for idx, (reg, plab) in enumerate(zip(region_order, panel_labels)):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        cf = draw_performance_background(ax)
        if cf_for_cbar is None:
            cf_for_cbar = cf

        for p in products:
            pod, far = region_stats.get(reg, {}).get(p, (np.nan, np.nan))
            if np.isnan(pod) or np.isnan(far):
                continue
            sr = 1.0 - far
            ax.scatter(
                sr, pod,
                marker=marker_map[p],
                color=color_map[p],
                s=marker_size,
                edgecolors="k",
                linewidths=0.6,
                zorder=5,
            )

        ax.text(
            0.02, 0.95, f"{plab} {reg}",
            transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            ha="left", va="top",
        )

    # 删除多余子图
    if n_region < nrows * ncols:
        for k in range(n_region, nrows * ncols):
            rr = k // ncols
            cc = k % ncols
            fig.delaxes(axes[rr, cc])

    # legend
    handles = []
    for p in products:
        h = plt.Line2D(
            [], [], linestyle="none",
            marker=marker_map[p], markersize=6,
            markerfacecolor=color_map[p], markeredgecolor="k",
            label=p,
        )
        handles.append(h)

    fig.legend(
        handles, products,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(products), 6),
        frameon=False,
        fontsize=8,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    # colorbar
    cax = fig.add_axes([0.92, 0.20, 0.02, 0.60])
    cbar = fig.colorbar(cf_for_cbar, cax=cax)
    cbar.set_label("Critical Success Index (CSI)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, y=0.98)

    plt.tight_layout(rect=[0.05, 0.08, 0.90, 0.96])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    return fig, axes


def save_performance_diagrams_regions_by_group(
    region_group_stats: dict,
    products: List[str],
    *,
    group_keys: List,                 # 例如 ["Spring","Summer","Autumn","Winter"] 或 [1..12]
    region_order: List[str],
    out_dir: str | Path,
    filename_fmt: str = "performance_regions_{key}.png",
    ncols: int = 3,
    dpi: int = 600,
    panel_size: Tuple[float, float] = (4.5, 4.0),
    title_fmt: str = "Performance diagram by regions: {key}",
):
    """
    把 region_group_stats[region][key][product] -> (POD,FAR)
    组装成每个 key 一张“分区多面板”图，并批量保存。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in group_keys:
        # 组装成 region_stats[region][product]
        region_stats = {}
        for r in region_order:
            d = region_group_stats.get(r, {})
            region_stats[r] = d.get(key, {})

        out_path = out_dir / filename_fmt.format(key=key)
        title = title_fmt.format(key=key)

        plot_performance_diagram_regions(
            region_stats,
            products,
            region_order=region_order,
            ncols=ncols,
            dpi=dpi,
            panel_size=panel_size,
            out_path=out_path,
            title=title,
        )

