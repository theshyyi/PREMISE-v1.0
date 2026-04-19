
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from .io import infer_main_var, open_nc_dataset
from .metrics import bias, mae, rmse, corr, kge, nse, pod, far, csi, fbias, hss
from .spatial import (
    bias_map, mae_map, rmse_map, corr_map, kge_map, nse_map,
    pod_map, far_map, csi_map, fbias_map, hss_map,
)
from .temporal import aggregate_xarray_time, subset_time

CONTINUOUS_METRICS = {"BIAS", "MAE", "RMSE", "CORR", "KGE", "NSE"}
EVENT_METRICS = {"POD", "FAR", "CSI", "FBIAS", "HSS"}


def _flatten_pair(obs: xr.DataArray, sim: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    sim_aligned, obs_aligned = xr.align(sim, obs, join="inner")
    o = obs_aligned.values.ravel()
    s = sim_aligned.values.ravel()
    mask = np.isfinite(o) & np.isfinite(s)
    return o[mask], s[mask]


def evaluate_grid_pair(
    obs: xr.DataArray,
    sim: xr.DataArray,
    *,
    threshold: float | None = None,
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    metrics = [m.upper() for m in (metrics or ["BIAS", "MAE", "RMSE", "CORR", "KGE", "NSE"])]
    o, s = _flatten_pair(obs, sim)
    if o.size == 0:
        return {m: np.nan for m in metrics}
    out: dict[str, float] = {}
    for m in metrics:
        if m == "BIAS":
            out[m] = bias(o, s)
        elif m == "MAE":
            out[m] = mae(o, s)
        elif m == "RMSE":
            out[m] = rmse(o, s)
        elif m == "CORR":
            out[m] = corr(o, s)
        elif m == "KGE":
            out[m] = kge(o, s)
        elif m == "NSE":
            out[m] = nse(o, s)
        elif m == "POD":
            out[m] = np.nan if threshold is None else pod(o, s, threshold)
        elif m == "FAR":
            out[m] = np.nan if threshold is None else far(o, s, threshold)
        elif m == "CSI":
            out[m] = np.nan if threshold is None else csi(o, s, threshold)
        elif m == "FBIAS":
            out[m] = np.nan if threshold is None else fbias(o, s, threshold)
        elif m == "HSS":
            out[m] = np.nan if threshold is None else hss(o, s, threshold)
        else:
            raise ValueError(f"不支持的指标: {m}")
    return out


def evaluate_grid_by_group(
    obs: xr.DataArray,
    sim: xr.DataArray,
    *,
    group_by: str = "month",
    threshold: float | None = None,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    sim, obs = xr.align(sim, obs, join="inner")
    if group_by == "month":
        group_obj = obs.groupby("time.month")
        label_name = "month"
    elif group_by == "season":
        group_obj = obs.groupby("time.season")
        label_name = "season"
    elif group_by == "year":
        group_obj = obs.groupby("time.year")
        label_name = "year"
    else:
        raise ValueError("group_by 仅支持 month / season / year")

    rows = []
    for label, obs_g in group_obj:
        sim_g = sim.sel(time=obs_g["time"])
        row = evaluate_grid_pair(obs_g, sim_g, threshold=threshold, metrics=metrics)
        row[label_name] = label.item() if hasattr(label, "item") else label
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_spatial_metric_dataset(
    obs: xr.DataArray,
    sim: xr.DataArray,
    *,
    metrics: Iterable[str] | None = None,
    threshold: float = 1.0,
) -> xr.Dataset:
    metrics = [m.upper() for m in (metrics or ["BIAS", "MAE", "RMSE", "CORR", "KGE", "NSE"])]
    sim, obs = xr.align(sim, obs, join="inner")
    data_vars = {}

    for m in metrics:
        if m == "BIAS":
            data_vars[m] = bias_map(obs, sim)
        elif m == "MAE":
            data_vars[m] = mae_map(obs, sim)
        elif m == "RMSE":
            data_vars[m] = rmse_map(obs, sim)
        elif m == "CORR":
            data_vars[m] = corr_map(obs, sim)
        elif m == "KGE":
            data_vars[m] = kge_map(obs, sim)
        elif m == "NSE":
            data_vars[m] = nse_map(obs, sim)
        elif m == "POD":
            data_vars[m] = pod_map(obs, sim, threshold)
        elif m == "FAR":
            data_vars[m] = far_map(obs, sim, threshold)
        elif m == "CSI":
            data_vars[m] = csi_map(obs, sim, threshold)
        elif m == "FBIAS":
            data_vars[m] = fbias_map(obs, sim, threshold)
        elif m == "HSS":
            data_vars[m] = hss_map(obs, sim, threshold)
        else:
            raise ValueError(f"不支持的指标: {m}")
    return xr.Dataset(data_vars)


def run_grid_evaluation(
    *,
    obs_path: str | Path,
    sim_path: str | Path,
    obs_var: str | None = None,
    sim_var: str | None = None,
    time_range: tuple[str, str] | None = None,
    time_scale: str = "native",
    time_agg: str = "sum",
    overall_metrics: Iterable[str] | None = None,
    spatial_metrics: Iterable[str] | None = None,
    group_by: str | None = None,
    threshold: float = 1.0,
    out_table_csv: str | Path | None = None,
    out_group_csv: str | Path | None = None,
    out_spatial_nc: str | Path | None = None,
) -> dict:
    ds_obs = open_nc_dataset(obs_path)
    ds_sim = open_nc_dataset(sim_path)
    try:
        obs_name = infer_main_var(ds_obs, obs_var)
        sim_name = infer_main_var(ds_sim, sim_var)
        obs = ds_obs[obs_name]
        sim = ds_sim[sim_name]

        if time_range is not None:
            obs = subset_time(obs, start=time_range[0], end=time_range[1])
            sim = subset_time(sim, start=time_range[0], end=time_range[1])

        obs = aggregate_xarray_time(obs, scale=time_scale, agg=time_agg)
        sim = aggregate_xarray_time(sim, scale=time_scale, agg=time_agg)
        sim, obs = xr.align(sim, obs, join="inner")

        overall = evaluate_grid_pair(obs, sim, threshold=threshold, metrics=overall_metrics)
        overall_df = pd.DataFrame([overall])

        grouped_df = None
        if group_by and group_by != "none":
            grouped_df = evaluate_grid_by_group(obs, sim, group_by=group_by, threshold=threshold, metrics=overall_metrics)

        spatial_ds = None
        if spatial_metrics:
            spatial_ds = compute_spatial_metric_dataset(obs, sim, metrics=spatial_metrics, threshold=threshold)

        if out_table_csv is not None:
            out_table_csv = Path(out_table_csv)
            out_table_csv.parent.mkdir(parents=True, exist_ok=True)
            overall_df.to_csv(out_table_csv, index=False)
        if out_group_csv is not None and grouped_df is not None:
            out_group_csv = Path(out_group_csv)
            out_group_csv.parent.mkdir(parents=True, exist_ok=True)
            grouped_df.to_csv(out_group_csv, index=False)
        if out_spatial_nc is not None and spatial_ds is not None:
            out_spatial_nc = Path(out_spatial_nc)
            out_spatial_nc.parent.mkdir(parents=True, exist_ok=True)
            spatial_ds.to_netcdf(out_spatial_nc)

        return {"overall": overall_df, "grouped": grouped_df, "spatial": spatial_ds}
    finally:
        ds_obs.close()
        ds_sim.close()
