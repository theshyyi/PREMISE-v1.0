
from __future__ import annotations

from typing import Literal

import pandas as pd
import xarray as xr

TimeScale = Literal["native", "daily", "monthly", "yearly"]
GroupBy = Literal["none", "month", "season", "year"]


def subset_time(
    data,
    *,
    time_dim: str = "time",
    start: str | None = None,
    end: str | None = None,
):
    if time_dim not in data.coords and time_dim not in data.dims:
        return data
    if start is None and end is None:
        return data
    return data.sel({time_dim: slice(start, end)})


def aggregate_xarray_time(
    data: xr.Dataset | xr.DataArray,
    *,
    scale: TimeScale = "native",
    agg: str = "sum",
    time_dim: str = "time",
):
    if scale in {"native", "daily"}:
        return data
    if time_dim not in data.dims and time_dim not in data.coords:
        return data

    if scale == "monthly":
        freq = "MS"
    elif scale == "yearly":
        freq = "YS"
    else:
        raise ValueError(f"不支持的时间尺度: {scale}")

    resampler = data.resample({time_dim: freq})
    if agg == "sum":
        return resampler.sum(skipna=True)
    if agg == "mean":
        return resampler.mean(skipna=True)
    if agg == "max":
        return resampler.max(skipna=True)
    if agg == "min":
        return resampler.min(skipna=True)
    raise ValueError(f"不支持的聚合方式: {agg}")


def aggregate_table_time(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    value_col: str = "obs",
    station_cols: list[str] | None = None,
    scale: TimeScale = "native",
    agg: str = "sum",
) -> pd.DataFrame:
    if scale in {"native", "daily"}:
        return df.copy()

    if station_cols is None:
        station_cols = [c for c in ["station", "station_id", "lat", "lon"] if c in df.columns]

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])

    if scale == "monthly":
        out["_period_time"] = out[time_col].dt.to_period("M").dt.to_timestamp()
    elif scale == "yearly":
        out["_period_time"] = out[time_col].dt.to_period("Y").dt.to_timestamp()
    else:
        raise ValueError(f"不支持的时间尺度: {scale}")

    group_cols = station_cols + ["_period_time"]
    gb = out.groupby(group_cols, dropna=False)[value_col]
    if agg == "sum":
        vals = gb.sum().reset_index()
    elif agg == "mean":
        vals = gb.mean().reset_index()
    elif agg == "max":
        vals = gb.max().reset_index()
    elif agg == "min":
        vals = gb.min().reset_index()
    else:
        raise ValueError(f"不支持的聚合方式: {agg}")

    vals = vals.rename(columns={"_period_time": time_col, value_col: value_col})
    return vals


def add_group_column(df: pd.DataFrame, *, time_col: str = "time", group_by: GroupBy = "none") -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])

    if group_by == "none":
        out["_group"] = "all"
    elif group_by == "month":
        out["_group"] = out[time_col].dt.month
    elif group_by == "season":
        out["_group"] = out[time_col].dt.season
    elif group_by == "year":
        out["_group"] = out[time_col].dt.year
    else:
        raise ValueError(f"不支持的 group_by: {group_by}")
    return out
