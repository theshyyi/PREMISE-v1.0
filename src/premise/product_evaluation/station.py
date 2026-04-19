
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr

from .io import infer_main_var, open_nc_dataset, open_reference_table
from .metrics import bias, mae, rmse, corr, kge, nse, pod, far, csi, fbias, hss
from .temporal import aggregate_table_time, aggregate_xarray_time, add_group_column, subset_time


def extract_pixel_series(
    ds: xr.Dataset,
    var_name: str,
    lat: float,
    lon: float,
    method: str = "nearest",
) -> xr.DataArray:
    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")
    return ds[var_name].sel(lat=lat, lon=lon, method=method)


def build_point_pixel_df(
    station_group: pd.DataFrame,
    ds: xr.Dataset,
    var_name: str,
    *,
    time_col: str,
    lat_col: str,
    lon_col: str,
    value_col: str,
    extract_method: str = "nearest",
) -> pd.DataFrame:
    lat = float(station_group[lat_col].iloc[0])
    lon = float(station_group[lon_col].iloc[0])

    st_series = (
        station_group[[time_col, value_col]]
        .sort_values(time_col)
        .set_index(time_col)[value_col]
    )
    st_series.name = "obs"

    da_point = extract_pixel_series(ds, var_name, lat=lat, lon=lon, method=extract_method)
    sim_series = da_point.to_pandas().sort_index()
    sim_series.name = "sim"

    df_pp = pd.concat([st_series, sim_series], axis=1).dropna()
    if not df_pp.empty:
        df_pp = df_pp.reset_index().rename(columns={"index": "time", time_col: "time"})
    else:
        df_pp = pd.DataFrame(columns=["time", "obs", "sim"])
    return df_pp


def compute_series_metrics(df_pp: pd.DataFrame, *, threshold: float, metrics: Iterable[str] | None = None) -> dict[str, float]:
    metrics = [m.upper() for m in (metrics or ["BIAS", "MAE", "RMSE", "CORR", "KGE", "NSE"])]
    if df_pp.empty:
        return {m: float("nan") for m in metrics}
    obs = df_pp["obs"].values
    sim = df_pp["sim"].values
    out = {}
    for m in metrics:
        if m == "BIAS":
            out[m] = bias(obs, sim)
        elif m == "MAE":
            out[m] = mae(obs, sim)
        elif m == "RMSE":
            out[m] = rmse(obs, sim)
        elif m == "CORR":
            out[m] = corr(obs, sim)
        elif m == "KGE":
            out[m] = kge(obs, sim)
        elif m == "NSE":
            out[m] = nse(obs, sim)
        elif m == "POD":
            out[m] = pod(obs, sim, threshold)
        elif m == "FAR":
            out[m] = far(obs, sim, threshold)
        elif m == "CSI":
            out[m] = csi(obs, sim, threshold)
        elif m == "FBIAS":
            out[m] = fbias(obs, sim, threshold)
        elif m == "HSS":
            out[m] = hss(obs, sim, threshold)
        else:
            raise ValueError(f"不支持的指标: {m}")
    return out


def run_station_evaluation(
    *,
    obs_table_path: str | Path,
    sim_nc_path: str | Path,
    obs_table_format: str | None = None,
    sim_var: str | None = None,
    time_range: tuple[str, str] | None = None,
    time_scale: str = "native",
    time_agg: str = "sum",
    station_col: str = "station",
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    value_col: str = "obs",
    extract_method: str = "nearest",
    metrics: Iterable[str] | None = None,
    group_by: str = "none",
    threshold: float = 1.0,
    out_station_csv: str | Path | None = None,
    out_group_csv: str | Path | None = None,
) -> dict:
    obs_df = open_reference_table(obs_table_path, fmt=obs_table_format, time_col=time_col)
    ds = open_nc_dataset(sim_nc_path)
    try:
        var_name = infer_main_var(ds, sim_var)

        if time_range is not None:
            obs_df = obs_df[(obs_df[time_col] >= pd.Timestamp(time_range[0])) & (obs_df[time_col] <= pd.Timestamp(time_range[1]))]
            ds_sel = subset_time(ds, start=time_range[0], end=time_range[1])
        else:
            ds_sel = ds

        obs_df = aggregate_table_time(obs_df, time_col=time_col, value_col=value_col, scale=time_scale, agg=time_agg)
        ds_sel = aggregate_xarray_time(ds_sel, scale=time_scale, agg=time_agg)

        rows = []
        group_rows = []
        for station_id, group in obs_df.groupby(station_col):
            df_pp = build_point_pixel_df(
                group,
                ds_sel,
                var_name,
                time_col=time_col,
                lat_col=lat_col,
                lon_col=lon_col,
                value_col=value_col,
                extract_method=extract_method,
            )
            met = compute_series_metrics(df_pp, threshold=threshold, metrics=metrics)
            row = {
                "station": station_id,
                "lat": float(group[lat_col].iloc[0]),
                "lon": float(group[lon_col].iloc[0]),
                "n": int(len(df_pp)),
                **met,
            }
            rows.append(row)

            if group_by and group_by != "none" and not df_pp.empty:
                df_group = add_group_column(df_pp, time_col="time", group_by=group_by)
                for g, grp in df_group.groupby("_group"):
                    gmet = compute_series_metrics(grp[["time", "obs", "sim"]], threshold=threshold, metrics=metrics)
                    group_rows.append({
                        "station": station_id,
                        "group": g,
                        "lat": float(group[lat_col].iloc[0]),
                        "lon": float(group[lon_col].iloc[0]),
                        "n": int(len(grp)),
                        **gmet,
                    })

        station_df = pd.DataFrame(rows)
        grouped_df = pd.DataFrame(group_rows) if group_rows else None

        if out_station_csv is not None:
            out_station_csv = Path(out_station_csv)
            out_station_csv.parent.mkdir(parents=True, exist_ok=True)
            station_df.to_csv(out_station_csv, index=False)
        if out_group_csv is not None and grouped_df is not None:
            out_group_csv = Path(out_group_csv)
            out_group_csv.parent.mkdir(parents=True, exist_ok=True)
            grouped_df.to_csv(out_group_csv, index=False)

        return {"station": station_df, "grouped": grouped_df}
    finally:
        ds.close()
