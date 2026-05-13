from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from ..core.netcdf import to_netcdf as _save_netcdf


def _candidate_files(path: str | Path, glob_pattern: str = "*.tif") -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    files = sorted(list(p.glob(glob_pattern)) + list(p.glob(glob_pattern.replace(".tif", ".tiff"))))
    return sorted({f.resolve() for f in files})


def _centers(transform, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([transform * (c + 0.5, 0.5) for c in range(width)], dtype=float)[:, 0]
    ys = np.array([transform * (0.5, r + 0.5) for r in range(height)], dtype=float)[:, 1]
    return xs, ys


def _parse_time(name: str, regex_pattern: str | None) -> pd.Timestamp | None:
    patterns: Iterable[str]
    if regex_pattern:
        patterns = [regex_pattern]
    else:
        patterns = [r"(\d{8})", r"(\d{4}-\d{2}-\d{2})", r"(\d{6})"]
    for pat in patterns:
        m = re.search(pat, name)
        if not m:
            continue
        token = m.group(1)
        for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y%m"):
            try:
                return pd.to_datetime(token, format=fmt)
            except Exception:
                pass
    return None


def open_geotiff_stack(
    path: str | Path,
    *,
    glob_pattern: str = "*.tif",
    regex_pattern: str | None = None,
    var_name: str = "pr",
    units: str | None = None,
    nodata: float | None = None,
) -> xr.Dataset:
    files = _candidate_files(path, glob_pattern=glob_pattern)
    if not files:
        raise FileNotFoundError(f"No GeoTIFF files found under {path}")

    data_list = []
    times = []
    lon = lat = None
    for fp in files:
        with rasterio.open(fp) as src:
            arr = src.read(1).astype("float32")
            nodata_value = src.nodata if nodata is None else nodata
            if nodata_value is not None:
                arr = np.where(arr == nodata_value, np.nan, arr)
            xs, ys = _centers(src.transform, src.width, src.height)
            lon = xs
            lat = ys
            data_list.append(arr)
            times.append(_parse_time(fp.name, regex_pattern))

    stack = np.stack(data_list, axis=0)
    dims = ("time", "lat", "lon") if len(files) > 1 else ("lat", "lon")
    if len(files) > 1:
        time_values = pd.to_datetime(times) if any(t is not None for t in times) else np.arange(len(files))
        da = xr.DataArray(stack, dims=dims, coords={"time": time_values, "lat": lat, "lon": lon}, name=var_name)
    else:
        da = xr.DataArray(stack[0], dims=dims, coords={"lat": lat, "lon": lon}, name=var_name)
    if units is not None:
        da.attrs["units"] = units
    return da.to_dataset(name=var_name)


def geotiff_to_monthly_netcdf(
    input_path: str | Path,
    out_nc: str | Path,
    *,
    glob_pattern: str = "*.tif",
    regex_pattern: str | None = None,
    var_name: str = "pr",
    units: str | None = None,
    nodata: float | None = None,
    reducer: str = "sum",
    comp_level: int = 4,
) -> str:
    ds = open_geotiff_stack(
        input_path,
        glob_pattern=glob_pattern,
        regex_pattern=regex_pattern,
        var_name=var_name,
        units=units,
        nodata=nodata,
    )
    if "time" in ds.dims:
        if not np.issubdtype(ds["time"].dtype, np.datetime64):
            raise ValueError("Monthly aggregation requires parseable daily timestamps in filenames.")
        if reducer == "sum":
            ds = ds.resample(time="MS").sum(keep_attrs=True)
        elif reducer == "mean":
            ds = ds.resample(time="MS").mean(keep_attrs=True)
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
    encoding = {
        var_name: {"zlib": True, "complevel": int(comp_level)} if ds[var_name].dtype.kind in "fiu" else {}
    }
    return _save_netcdf(ds, out_nc, encoding=encoding)
