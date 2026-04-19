
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import xarray as xr


def open_nc_dataset(path: str | Path, *, chunks: dict | None = None) -> xr.Dataset:
    return xr.open_dataset(Path(path), chunks=chunks)


def infer_main_var(ds: xr.Dataset, preferred: str | None = None) -> str:
    if preferred is not None:
        if preferred not in ds.data_vars:
            raise KeyError(f"变量 {preferred!r} 不在数据集中。可用变量: {list(ds.data_vars)}")
        return preferred
    for name, da in ds.data_vars.items():
        if "time" in da.dims:
            return name
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"无法自动识别主变量。可用变量: {list(ds.data_vars)}")


def open_reference_table(
    path: str | Path,
    *,
    fmt: Literal["csv", "xlsx"] | None = None,
    time_col: str = "time",
) -> pd.DataFrame:
    path = Path(path)
    if fmt is None:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            fmt = "csv"
        elif suffix in {".xlsx", ".xls"}:
            fmt = "xlsx"
        else:
            raise ValueError(f"不支持的表格格式: {path.suffix}")

    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "xlsx":
        df = pd.read_excel(path)
    else:
        raise ValueError(f"不支持的表格格式: {fmt}")

    if time_col not in df.columns:
        raise KeyError(f"表格中未找到时间列 {time_col!r}")
    df[time_col] = pd.to_datetime(df[time_col])
    return df
