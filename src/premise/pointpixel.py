# -*- coding: utf-8 -*-

"""
pointpixel
==========

Station–grid point–pixel comparison workflows:

- Read station daily precipitation table (CSV)
- For each station, extract nearest grid-cell time series from NetCDF products
- Build obs vs sim time series per station
- Compute continuous metrics (BIAS, MAE, RMSE, CORR, KGE)
  and detection metrics (POD, FAR, CSI, FBIAS)
- Optional: compute per-station metrics by month / season
- Save per-station CSVs and product-level metrics

Typical usage:
    from premise.pointpixel import run_point_pixel_for_all_products
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from .metrics import (
    bias,
    mae,
    rmse,
    corr,
    kge,
    pod,
    far,
    csi,
    fbias,
)


_all_ = [
    "compute_station_metrics",
    "compute_station_metrics_by_month",
    "compute_station_metrics_by_season",
    "run_point_pixel_for_product",
    "run_point_pixel_for_all_products",
]

def load_station_table(
    csv_path: str,
    time_col: str = "date",
    time_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read station CSV and parse time column to datetime.
    """
    df = pd.read_csv(csv_path)
    if time_format is not None:
        df[time_col] = pd.to_datetime(df[time_col], format=time_format)
    else:
        df[time_col] = pd.to_datetime(df[time_col])
    return df


def extract_pixel_series(
    ds: xr.Dataset,
    var_name: str,
    lat: float,
    lon: float,
    method: str = "nearest",
) -> xr.DataArray:
    """
    Extract a grid-cell time series at the given lat/lon.
    """
    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")
    da = ds[var_name]
    da_point = da.sel(lat=lat, lon=lon, method=method)
    return da_point


def build_point_pixel_df(
    station_group: pd.DataFrame,
    ds: xr.Dataset,
    var_name: str,
    station_id: Any,
    *,
    time_col: str,
    lat_col: str,
    lon_col: str,
    precip_col: str,
) -> pd.DataFrame:
    """
    Build obs vs sim time series table for a single station.
    """
    lat = float(station_group[lat_col].iloc[0])
    lon = float(station_group[lon_col].iloc[0])

    st_series = (
        station_group[[time_col, precip_col]]
        .sort_values(time_col)
        .set_index(time_col)[precip_col]
    )
    st_series.name = "obs"

    da_point = extract_pixel_series(ds, var_name, lat=lat, lon=lon, method="nearest")
    sim_series = da_point.to_pandas().sort_index()
    sim_series.name = "sim"

    df_pp = pd.concat([st_series, sim_series], axis=1).dropna()

    if not df_pp.empty:
        df_pp = df_pp.reset_index().rename(columns={"index": "time", time_col: "time"})
    else:
        df_pp = pd.DataFrame(columns=["time", "obs", "sim"])

    return df_pp


# ========== 站点级指标计算（整体 / 月 / 季节） ==========

def compute_station_metrics(
    df_pp: pd.DataFrame,
    threshold: float,
) -> Dict[str, float]:
    """
    基于 point–pixel 表 (time, obs, sim) 计算整体指标。

    返回:
        BIAS, MAE, RMSE, CORR, KGE, POD, FAR, CSI, FBIAS
    """
    if df_pp.empty:
        return {
            "BIAS": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "CORR": np.nan,
            "KGE": np.nan,
            "POD": np.nan,
            "FAR": np.nan,
            "CSI": np.nan,
            "FBIAS": np.nan,
        }

    obs_vals = df_pp["obs"].values
    sim_vals = df_pp["sim"].values

    out = {
        "BIAS": bias(obs_vals, sim_vals),
        "MAE": mae(obs_vals, sim_vals),
        "RMSE": rmse(obs_vals, sim_vals),
        "CORR": corr(obs_vals, sim_vals),
        "KGE": kge(obs_vals, sim_vals),
        "POD": pod(obs_vals, sim_vals, threshold),
        "FAR": far(obs_vals, sim_vals, threshold),
        "CSI": csi(obs_vals, sim_vals, threshold),
        "FBIAS": fbias(obs_vals, sim_vals, threshold),
    }
    return out


def compute_station_metrics_by_month(
    df_pp: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    对单个站点的 point–pixel 结果按月计算指标。

    返回 DataFrame:
        month, BIAS, MAE, RMSE, CORR, KGE, POD, FAR, CSI, FBIAS
    """
    if df_pp.empty:
        return pd.DataFrame()

    df = df_pp.copy()
    df["month"] = df["time"].dt.month

    rows: List[Dict[str, Any]] = []
    for m, grp in df.groupby("month"):
        met = compute_station_metrics(grp, threshold=threshold)
        met["month"] = int(m)
        rows.append(met)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def compute_station_metrics_by_season(
    df_pp: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    对单个站点的 point–pixel 结果按季节计算指标 (DJF/MAM/JJA/SON)。

    返回 DataFrame:
        season, BIAS, MAE, RMSE, CORR, KGE, POD, FAR, CSI, FBIAS
    """
    if df_pp.empty:
        return pd.DataFrame()

    df = df_pp.copy()
    df["month"] = df["time"].dt.month

    season_map = {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }
    df["season"] = df["month"].map(season_map)

    rows: List[Dict[str, Any]] = []
    for s, grp in df.groupby("season"):
        met = compute_station_metrics(grp, threshold=threshold)
        met["season"] = str(s)
        rows.append(met)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


# ========== 批量运行（与之前类似，只是多写了几个指标列） ==========

def run_point_pixel_for_product(
    station_df: pd.DataFrame,
    nc_path: str,
    product_name: Optional[str] = None,
    *,
    var_name: str = "pr",
    station_col: str = "station",
    time_col: str = "date",
    lat_col: str = "lat",
    lon_col: str = "lon",
    precip_col: str = "pr",
    threshold: float = 1.0,
    out_root_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run point–pixel comparison for all stations for a single product.

    Optionally saves:
        out_root_dir/{PRODUCT}/
            ├─ {PRODUCT}_PointPixel_{station}.csv
            └─ PointPixel_metrics.csv
    """
    fname = os.path.basename(nc_path)
    if product_name is None:
        product_name = fname.split(".TIMEFIX")[0]

    print(f"[premise] Point–pixel for product: {product_name}")

    ds = xr.open_dataset(nc_path)

    product_dir = None
    if out_root_dir is not None:
        product_dir = os.path.join(out_root_dir, product_name)
        os.makedirs(product_dir, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []

    for station_id, group in station_df.groupby(station_col):
        df_pp = build_point_pixel_df(
            station_group=group,
            ds=ds,
            var_name=var_name,
            station_id=station_id,
            time_col=time_col,
            lat_col=lat_col,
            lon_col=lon_col,
            precip_col=precip_col,
        )

        if df_pp.empty:
            print(f"  - WARNING: station {station_id} has no overlapping data, skip.")
            continue

        met = compute_station_metrics(df_pp, threshold=threshold)

        lat_val = float(group[lat_col].iloc[0])
        lon_val = float(group[lon_col].iloc[0])

        metrics_rows.append(
            {
                "product": product_name,
                "station": station_id,
                "lat": lat_val,
                "lon": lon_val,
                "threshold": threshold,
                **met,
            }
        )

        if product_dir is not None:
            out_csv = os.path.join(
                product_dir,
                f"{product_name}_PointPixel_{station_id}.csv",
            )
            df_pp.to_csv(out_csv, index=False)

    ds.close()

    metrics_df = pd.DataFrame(metrics_rows)
    if product_dir is not None and not metrics_df.empty:
        metrics_csv = os.path.join(product_dir, "PointPixel_metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"  - Metrics saved to: {metrics_csv}")

    return metrics_df


def run_point_pixel_for_all_products(
    station_csv: str,
    nc_dir: str,
    out_root_dir: str,
    *,
    var_name: str = "pr",
    station_col: str = "station",
    time_col: str = "date",
    lat_col: str = "lat",
    lon_col: str = "lon",
    precip_col: str = "pr",
    threshold: float = 1.0,
    pattern: str = "*.TIMEFIX.daily.CHINA.nc",
) -> pd.DataFrame:
    """
    Run point–pixel comparison for all products in a directory.

    Outputs:
        - {PRODUCT}/PointPixel_metrics.csv (per product)
        - PointPixel_metrics_all_products.csv (summary)
    """
    station_df = load_station_table(
        station_csv,
        time_col=time_col,
        time_format=None,
    )

    nc_paths = sorted(glob.glob(os.path.join(nc_dir, pattern)))
    if not nc_paths:
        raise FileNotFoundError(f"No files matched pattern {pattern} in directory {nc_dir}")

    all_rows: List[pd.DataFrame] = []

    for nc_path in nc_paths:
        product_metrics = run_point_pixel_for_product(
            station_df=station_df,
            nc_path=nc_path,
            product_name=None,
            var_name=var_name,
            station_col=station_col,
            time_col=time_col,
            lat_col=lat_col,
            lon_col=lon_col,
            precip_col=precip_col,
            threshold=threshold,
            out_root_dir=out_root_dir,
        )
        if not product_metrics.empty:
            all_rows.append(product_metrics)

    if not all_rows:
        raise RuntimeError("No metrics computed for any product.")

    all_metrics_df = pd.concat(all_rows, ignore_index=True)

    summary_csv = os.path.join(out_root_dir, "PointPixel_metrics_all_products.csv")
    all_metrics_df.to_csv(summary_csv, index=False)
    print(f"[premise] All-product metrics summary saved to: {summary_csv}")

    return all_metrics_df
