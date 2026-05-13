from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr


InterpMethod = Literal["nearest", "linear", "bilinear"]
AggMethod = Literal["mean", "sum", "max", "min"]
ResampleMethod = Literal["nearest", "linear", "bilinear", "mean", "sum", "max", "min"]


def _infer_lat_lon_names(ds: xr.Dataset | xr.DataArray, lat_name: str | None = None, lon_name: str | None = None) -> tuple[str, str]:
    if lat_name is None:
        for cand in ["lat", "latitude", "y"]:
            if cand in ds.coords or cand in ds.dims:
                lat_name = cand
                break
    if lon_name is None:
        for cand in ["lon", "longitude", "x"]:
            if cand in ds.coords or cand in ds.dims:
                lon_name = cand
                break

    if lat_name is None or lon_name is None:
        raise ValueError("无法识别纬度/经度坐标名称，请显式提供 lat_name 和 lon_name。")

    return lat_name, lon_name


def _get_resolution(coord: xr.DataArray | np.ndarray) -> float:
    values = np.asarray(coord, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("坐标必须是一维且长度>=2，才能推断分辨率。")
    diffs = np.abs(np.diff(values))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError("无法从坐标推断分辨率。")
    return float(np.median(diffs))


def _build_target_coords(
    ds: xr.Dataset | xr.DataArray,
    target_resolution: float,
    lat_name: str,
    lon_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(ds[lat_name].values, dtype=float)
    lon = np.asarray(ds[lon_name].values, dtype=float)

    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))

    lat_asc = lat[0] < lat[-1]
    lon_asc = lon[0] < lon[-1]

    new_lat = np.arange(lat_min, lat_max + target_resolution * 0.5, target_resolution)
    new_lon = np.arange(lon_min, lon_max + target_resolution * 0.5, target_resolution)

    if not lat_asc:
        new_lat = new_lat[::-1]
    if not lon_asc:
        new_lon = new_lon[::-1]

    return new_lat, new_lon


def _interp_resample(
    ds: xr.Dataset | xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    method: InterpMethod,
    lat_name: str,
    lon_name: str,
) -> xr.Dataset | xr.DataArray:
    actual_method = "linear" if method == "bilinear" else method
    return ds.interp(
        coords={lat_name: target_lat, lon_name: target_lon},
        method=actual_method,
    )


def _check_aligned_coarsen(
    ds: xr.Dataset | xr.DataArray,
    target_resolution: float,
    lat_name: str,
    lon_name: str,
) -> tuple[int, int]:
    src_lat_res = _get_resolution(ds[lat_name].values)
    src_lon_res = _get_resolution(ds[lon_name].values)

    lat_ratio = target_resolution / src_lat_res
    lon_ratio = target_resolution / src_lon_res

    lat_factor = int(round(lat_ratio))
    lon_factor = int(round(lon_ratio))

    if lat_factor < 1 or lon_factor < 1:
        raise ValueError(
            f"聚合重采样只适用于变粗分辨率。当前源分辨率约为 "
            f"({src_lat_res}, {src_lon_res})，目标分辨率为 {target_resolution}。"
        )

    if not np.isclose(lat_ratio, lat_factor, atol=1e-6) or not np.isclose(lon_ratio, lon_factor, atol=1e-6):
        raise ValueError(
            f"当前 target_resolution={target_resolution} 不是源分辨率的整数倍，"
            f"无法安全执行 mean/sum/max/min 聚合重采样。"
        )

    return lat_factor, lon_factor


def _agg_resample(
    ds: xr.Dataset | xr.DataArray,
    target_resolution: float,
    method: AggMethod,
    lat_name: str,
    lon_name: str,
) -> xr.Dataset | xr.DataArray:
    lat_factor, lon_factor = _check_aligned_coarsen(ds, target_resolution, lat_name, lon_name)

    coarsened = ds.coarsen(
        {lat_name: lat_factor, lon_name: lon_factor},
        boundary="trim",
    )

    if method == "mean":
        out = coarsened.mean()
    elif method == "sum":
        out = coarsened.sum()
    elif method == "max":
        out = coarsened.max()
    elif method == "min":
        out = coarsened.min()
    else:
        raise ValueError(f"不支持的聚合重采样方法: {method}")

    return out


def spatial_resample(
    ds: xr.Dataset | xr.DataArray,
    *,
    target_resolution: float | None = None,
    target_grid_path: str | Path | None = None,
    method: ResampleMethod = "linear",
    lat_name: str | None = None,
    lon_name: str | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    空间重采样，不依赖 xESMF。

    支持方法：
    - nearest
    - linear
    - bilinear   -> 规则经纬网格下等价于 linear
    - mean
    - sum
    - max
    - min

    说明：
    1. nearest / linear / bilinear 使用 xarray.interp
    2. mean / sum / max / min 使用 xarray.coarsen
       因此要求 target_resolution 是源分辨率的整数倍，且适合从细网格到粗网格
    """
    lat_name, lon_name = _infer_lat_lon_names(ds, lat_name, lon_name)

    if (target_resolution is None) == (target_grid_path is None):
        raise ValueError("target_resolution 和 target_grid_path 必须二选一。")

    if target_grid_path is not None:
        target_grid = xr.open_dataset(target_grid_path)
        try:
            tgt_lat_name, tgt_lon_name = _infer_lat_lon_names(target_grid, lat_name, lon_name)
            target_lat = np.asarray(target_grid[tgt_lat_name].values, dtype=float)
            target_lon = np.asarray(target_grid[tgt_lon_name].values, dtype=float)
        finally:
            target_grid.close()

        if method in {"mean", "sum", "max", "min"}:
            raise ValueError("当使用 target_grid_path 时，当前版本仅支持 nearest/linear/bilinear，不支持聚合型重采样。")

        return _interp_resample(ds, target_lat, target_lon, method, lat_name, lon_name)

    # target_resolution 路径
    assert target_resolution is not None

    if method in {"nearest", "linear", "bilinear"}:
        target_lat, target_lon = _build_target_coords(ds, float(target_resolution), lat_name, lon_name)
        return _interp_resample(ds, target_lat, target_lon, method, lat_name, lon_name)

    if method in {"mean", "sum", "max", "min"}:
        return _agg_resample(ds, float(target_resolution), method, lat_name, lon_name)

    raise ValueError(f"不支持的重采样方法: {method}")


def resample_to_resolution(
    ds: xr.Dataset | xr.DataArray,
    resolution: float,
    method: ResampleMethod = "linear",
    lat_name: str | None = None,
    lon_name: str | None = None,
) -> xr.Dataset | xr.DataArray:
    return spatial_resample(
        ds,
        target_resolution=resolution,
        method=method,
        lat_name=lat_name,
        lon_name=lon_name,
    )


def resample_to_grid(
    ds: xr.Dataset | xr.DataArray,
    target_grid_path: str | Path,
    method: Literal["nearest", "linear", "bilinear"] = "linear",
    lat_name: str | None = None,
    lon_name: str | None = None,
) -> xr.Dataset | xr.DataArray:
    return spatial_resample(
        ds,
        target_grid_path=target_grid_path,
        method=method,
        lat_name=lat_name,
        lon_name=lon_name,
    )