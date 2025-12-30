# -*- coding: utf-8 -*-

"""
evaluation
==========

常规网格-网格评估封装：

- evaluate_grid_pair       : 整体统计
- evaluate_grid_by_month   : 按月统计
- evaluate_grid_by_season  : 按季节统计 (DJF/MAM/JJA/SON)

全部调用 premise.metrics 中的 BIAS/MAE/RMSE/CORR/KGE + POD/FAR/CSI/FBIAS。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Optional

import geopandas as gpd
import regionmask



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

__all__ = [
    "evaluate_grid_pair",
    "evaluate_grid_by_month",
    "evaluate_grid_by_season",
    "compute_pod_far_xr",
    "compute_pod_far_temporal_from_directory",
    "compute_pod_far_by_elevation_from_directory",
    "compute_pod_far_by_regions_from_directory",
]
def _flatten_pair(
    obs: xr.DataArray,
    sim: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对齐 time 维度并展平为 1D 数组。
    """
    sim_aligned, obs_aligned = xr.align(sim, obs, join="inner")
    o = obs_aligned.values.ravel()
    s = sim_aligned.values.ravel()
    mask = np.isfinite(o) & np.isfinite(s)
    return o[mask], s[mask]


def evaluate_grid_pair(
    obs: xr.DataArray,
    sim: xr.DataArray,
    *,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    对单一 obs/sim 网格场整体计算常用指标。
    """
    o, s = _flatten_pair(obs, sim)
    if o.size == 0:
        return {k: np.nan for k in ["BIAS", "MAE", "RMSE", "CORR", "KGE",
                                    "POD", "FAR", "CSI", "FBIAS"]}

    out = {
        "BIAS": bias(o, s),
        "MAE": mae(o, s),
        "RMSE": rmse(o, s),
        "CORR": corr(o, s),
        "KGE": kge(o, s),
    }

    if threshold is not None:
        out.update(
            {
                "POD": pod(o, s, threshold),
                "FAR": far(o, s, threshold),
                "CSI": csi(o, s, threshold),
                "FBIAS": fbias(o, s, threshold),
            }
        )
    else:
        out.update(
            {
                "POD": np.nan,
                "FAR": np.nan,
                "CSI": np.nan,
                "FBIAS": np.nan,
            }
        )

    return out


def evaluate_grid_by_month(
    obs: xr.DataArray,
    sim: xr.DataArray,
    *,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    按月对 obs/sim 计算各类指标，返回 DataFrame:

        month, BIAS, MAE, RMSE, CORR, KGE, POD, FAR, CSI, FBIAS
    """
    sim, obs = xr.align(sim, obs, join="inner")

    records = []
    for month, obs_m in obs.groupby("time.month"):
        sim_m = sim.sel(time=obs_m["time"])

        metrics = evaluate_grid_pair(obs_m, sim_m, threshold=threshold)
        metrics["month"] = int(month)
        records.append(metrics)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("month").reset_index(drop=True)
    return df


def evaluate_grid_by_season(
    obs: xr.DataArray,
    sim: xr.DataArray,
    *,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    按季节对 obs/sim 计算各类指标，返回 DataFrame:

        season, BIAS, MAE, RMSE, CORR, KGE, POD, FAR, CSI, FBIAS

    其中 season 为 'DJF', 'MAM', 'JJA', 'SON'。
    """
    sim, obs = xr.align(sim, obs, join="inner")

    records = []
    for season, obs_s in obs.groupby("time.season"):
        sim_s = sim.sel(time=obs_s["time"])

        metrics = evaluate_grid_pair(obs_s, sim_s, threshold=threshold)
        metrics["season"] = str(season)
        records.append(metrics)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("season").reset_index(drop=True)
    return df


def detect_precip_var(ds: xr.Dataset) -> str:
    """
    自动识别 time/lat/lon 的降水变量名。
    若用户已知变量名，也可以直接传入，不用此函数。
    """
    for v, da in ds.data_vars.items():
        if {"time", "lat", "lon"}.issubset(set(da.dims)):
            return v
    raise ValueError("未找到同时包含 time/lat/lon 维度的降水变量，请检查数据。")


def build_domain_mask_from_shp(
        da_like: xr.DataArray,
        shp_path: str | Path | None,
) -> xr.DataArray | None:
    """
    根据 shapefile 在 da_like 的网格上生成布尔掩膜（True=在多边形内）。

    - 不局限于中国，任何区域 shp 都可以
    - 如果 shp_path 为 None，则返回 None（不做掩膜）
    """
    if shp_path is None:
        return None

    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"shapefile 未找到: {shp_path}")

    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)

    regions = regionmask.from_geopandas(gdf)
    lon = da_like["lon"].values
    lat = da_like["lat"].values
    mask = regions.mask(lon, lat)  # (lat, lon)
    return mask.notnull()


def compute_pod_far_xr(
        obs: xr.DataArray,
        sim: xr.DataArray,
        threshold: float,
        mask: xr.DataArray | None = None,
) -> Tuple[float, float]:
    """
    基于 xarray 计算 POD 与 FAR。

    POD = hits / (hits + misses)
    FAR = false_alarm / (hits + false_alarm)

    obs, sim: (time, lat, lon)
    mask: (lat, lon) bool，True 表示参与统计的位置（可选）
    """
    obs, sim = xr.align(obs, sim, join="inner")

    if mask is not None:
        obs = obs.where(mask)
        sim = sim.where(mask)

    valid = np.isfinite(obs) & np.isfinite(sim)

    obs_evt = obs >= threshold
    sim_evt = sim >= threshold

    hits_da = ((obs_evt & sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True)
    miss_da = ((obs_evt & ~sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True)
    fa_da = ((~obs_evt & sim_evt) & valid).sum(dim=("time", "lat", "lon"), skipna=True)

    hits = float(hits_da)
    miss = float(miss_da)
    fa = float(fa_da)

    pod = np.nan
    far = np.nan
    if hits + miss > 0:
        pod = hits / (hits + miss)
    if hits + fa > 0:
        far = fa / (hits + fa)

    return pod, far


def _subset_by_months(da: xr.DataArray, months: Sequence[int]) -> xr.DataArray:
    """按月份列表子集数据。"""
    return da.sel(time=da["time"].dt.month.isin(months))


def compute_pod_far_temporal_from_directory(
    nc_dir: str | Path,
    ref_name: str,
    *,
    pattern: str = "*.TIMEFIX.daily.CHINA.nc",
    precip_var: str | None = None,
    shp_path: str | Path | None = None,
    threshold: float = 1.0,
    seasons: Mapping[str, Sequence[int]] | None = None,
    months: Sequence[int] | None = None,
    time_range: Tuple[str, str] | None = None,
) -> Tuple[
    Dict[str, Dict[str, Tuple[float, float]]],
    Dict[int, Dict[str, Tuple[float, float]]],
    List[str],
]:
    """
    遍历目录中的多个降水产品，计算相对于参考产品的 POD/FAR：

    - season_stats[season][product] = (POD, FAR)
    - month_stats[month][product]   = (POD, FAR)

    参数
    ----
    nc_dir : str or Path
        存放各产品 nc 文件的目录。
    ref_name : str
        参考产品名（文件名前缀），例如 "CMFDV2"。
    pattern : str, default "*.TIMEFIX.daily.CHINA.nc"
        用于匹配产品文件的 glob 模式。
    precip_var : str, optional
        若已知降水变量名可指定；否则自动检测。
    shp_path : str or Path, optional
        若提供，则根据该 shp 构建区域掩膜，只统计多边形内部。
    threshold : float
        降水事件阈值（mm/day）。
    seasons : Mapping[str, Sequence[int]], optional
        季节定义，如 {"Spring":[3,4,5], ...}；默认使用 (MAM/JJA/SON/DJF)。
    months : sequence[int], optional
        需要统计的月份，默认 1–12。
    time_range : (start, end), optional
        时间范围，例如 ("2000-01-01","2022-12-31")，不指定则用全时段。

    返回
    ----
    season_stats, month_stats, products
    """
    nc_dir = Path(nc_dir)
    paths = sorted(nc_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"目录 {nc_dir} 中未找到 {pattern} 文件。")

    ref_path = None
    product_paths: Dict[str, Path] = {}
    for p in paths:
        prod = p.name.split(".")[0]
        if prod == ref_name:
            ref_path = p
        else:
            product_paths[prod] = p

    if ref_path is None:
        raise FileNotFoundError(f"未找到参考产品 {ref_name} 对应的文件（匹配模式 {pattern}）")

    # 打开参考数据
    ref_ds = xr.open_dataset(ref_path, chunks={"time": 90})
    var = precip_var or detect_precip_var(ref_ds)
    ref_da = ref_ds[var]
    if time_range is not None:
        ref_da = ref_da.sel(time=slice(time_range[0], time_range[1]))

    # 区域掩膜
    domain_mask = build_domain_mask_from_shp(ref_da, shp_path)

    # 季节定义
    if seasons is None:
        seasons = {
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Autumn": [9, 10, 11],
            "Winter": [12, 1, 2],
        }
    season_order = list(seasons.keys())

    if months is None:
        months = list(range(1, 13))

    season_stats: Dict[str, Dict[str, Tuple[float, float]]] = {
        s: {} for s in season_order
    }
    month_stats: Dict[int, Dict[str, Tuple[float, float]]] = {
        m: {} for m in months
    }

    # 遍历每个产品
    for prod, path in product_paths.items():
        sim_ds = xr.open_dataset(path, chunks={"time": 90})
        sim_var = precip_var or detect_precip_var(sim_ds)
        sim_da = sim_ds[sim_var]
        if time_range is not None:
            sim_da = sim_da.sel(time=slice(time_range[0], time_range[1]))

        # 季节
        for sname, mon_list in seasons.items():
            obs_s = _subset_by_months(ref_da, mon_list)
            sim_s = _subset_by_months(sim_da, mon_list)
            pod, far = compute_pod_far_xr(obs_s, sim_s, threshold, domain_mask)
            season_stats[sname][prod] = (pod, far)

        # 月份
        for m in months:
            obs_m = _subset_by_months(ref_da, [m])
            sim_m = _subset_by_months(sim_da, [m])
            pod, far = compute_pod_far_xr(obs_m, sim_m, threshold, domain_mask)
            month_stats[m][prod] = (pod, far)

        sim_ds.close()

    ref_ds.close()
    products = list(product_paths.keys())
    return season_stats, month_stats, products


def _load_dem_on_grid(
    dem_nc: str | Path,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
    elev_var: str | None = None,
) -> xr.DataArray:
    """
    读取 DEM，并插值到目标网格 (target_lat, target_lon)。

    - 若 DEM 维度为 (y, x)，会自动重命名为 (lat, lon)。
    - elev_var 未指定时，自动寻找含 lat/lon 维度的第一个变量。
    """
    dem_nc = Path(dem_nc)
    dem_ds = xr.open_dataset(dem_nc)

    # dims 统一为 lat/lon
    if "y" in dem_ds.dims and "x" in dem_ds.dims:
        dem_ds = dem_ds.rename({"y": "lat", "x": "lon"})

    if elev_var is None:
        for v, da in dem_ds.data_vars.items():
            if {"lat", "lon"}.issubset(set(da.dims)):
                elev_var = v
                break
        if elev_var is None:
            raise ValueError("DEM 文件中未找到包含 lat/lon 维度的高程变量。")

    dem_da = dem_ds[elev_var].astype("float32")

    # 去掉多余维度，只保留 lat/lon
    for d in list(dem_da.dims):
        if d not in ("lat", "lon"):
            dem_da = dem_da.isel({d: 0})

    if (not np.array_equal(dem_da["lat"], target_lat)) or (
        not np.array_equal(dem_da["lon"], target_lon)
    ):
        dem_da = dem_da.interp(lat=target_lat, lon=target_lon, method="linear")

    dem_ds.close()
    return dem_da


def compute_pod_far_by_elevation_from_directory(
    nc_dir: str | Path,
    ref_name: str,
    *,
    dem_nc: str | Path,
    elev_bins: Sequence[Tuple[float, float]],
    threshold: float,
    year_start: int | None = None,
    year_end: int | None = None,
    pattern: str = "*.TIMEFIX.daily.CHINA.nc",
    precip_var: str | None = None,
    shp_path: str | Path | None = None,
) -> Tuple[
    List[str],
    Dict[str, Dict[str, Tuple[float, float]]],
    List[str],
]:
    """
    按海拔带统计多个产品的 POD/FAR。

    返回
    ----
    elev_labels : list[str]
        每个海拔带的标签，如 "0-200 m"。
    elev_stats  : dict[label][product] = (POD, FAR)
    products    : list[str]
        产品名列表（不含参考产品）。
    """
    nc_dir = Path(nc_dir)
    paths = sorted(nc_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"目录 {nc_dir} 中未找到 {pattern} 文件。")

    ref_path = None
    product_paths: Dict[str, Path] = {}
    for p in paths:
        prod = p.name.split(".")[0]
        if prod == ref_name:
            ref_path = p
        else:
            product_paths[prod] = p

    if ref_path is None:
        raise FileNotFoundError(f"未找到参考产品 {ref_name} 对应的文件。")

    # 打开参考
    ref_ds = xr.open_dataset(ref_path, chunks={"time": 90})
    var = precip_var or detect_precip_var(ref_ds)
    ref_da = ref_ds[var]
    if year_start is not None and year_end is not None:
        ref_da = ref_da.sel(
            time=slice(f"{year_start}-01-01", f"{year_end}-12-31")
        )

    # 区域掩膜（整个陆地区域）
    domain_mask = build_domain_mask_from_shp(ref_da, shp_path)

    # DEM 插值到参考网格
    dem_da = _load_dem_on_grid(
        dem_nc,
        ref_da["lat"],
        ref_da["lon"],
    )

    # 海拔带掩膜
    elev_labels: List[str] = []
    elev_masks: Dict[str, xr.DataArray] = {}
    for lo, hi in elev_bins:
        label = f"{lo}-{hi} m"
        elev_labels.append(label)
        m = (dem_da >= lo) & (dem_da < hi)
        if domain_mask is not None:
            m = m & domain_mask
        elev_masks[label] = m

    elev_stats: Dict[str, Dict[str, Tuple[float, float]]] = {
        lab: {} for lab in elev_labels
    }

    # 遍历各产品
    for prod, path in product_paths.items():
        sim_ds = xr.open_dataset(path, chunks={"time": 90})
        sim_var = precip_var or detect_precip_var(sim_ds)
        sim_da = sim_ds[sim_var]
        if year_start is not None and year_end is not None:
            sim_da = sim_da.sel(
                time=slice(f"{year_start}-01-01", f"{year_end}-12-31")
            )

        for lab in elev_labels:
            pod, far = compute_pod_far_xr(
                ref_da, sim_da, threshold, elev_masks[lab]
            )
            elev_stats[lab][prod] = (pod, far)

        sim_ds.close()

    ref_ds.close()
    products = list(product_paths.keys())
    return elev_labels, elev_stats, products

def build_region_masks_from_shp(
    da_like: xr.DataArray,
    shp_path: str | Path,
    region_field: str,
) -> Tuple[List[str], Dict[str, xr.DataArray]]:
    """
    基于 shp 的 region_field 字段，构建每个分区的 (lat, lon) 布尔掩膜。

    返回
    ----
    region_names : list[str]
    region_masks : dict[name] -> xr.DataArray(lat, lon) bool
    """
    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"shapefile 未找到: {shp_path}")

    gdf = gpd.read_file(shp_path)

    # 投影到 WGS84，保证与 lon/lat 网格一致
    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)

    if region_field not in gdf.columns:
        raise KeyError(f"shapefile 中未找到字段 '{region_field}'。")

    # 同名分区可能有多个多边形，先 dissolve 合并
    gdf2 = gdf[[region_field, "geometry"]].copy()
    gdf2[region_field] = gdf2[region_field].astype(str)
    gdf2 = gdf2.dissolve(by=region_field, as_index=True)

    region_names = gdf2.index.astype(str).tolist()
    geoms = list(gdf2.geometry.values)

    regions = regionmask.Regions(
        outlines=geoms,
        names=region_names,
        numbers=list(range(len(region_names))),
    )

    lon = da_like["lon"].values
    lat = da_like["lat"].values

    # (lat, lon) 的整数分区编号 mask，NaN 为不在任何多边形内
    rid = regions.mask(lon, lat)

    region_masks: Dict[str, xr.DataArray] = {}
    for i, name in enumerate(region_names):
        region_masks[name] = (rid == i)

    return region_names, region_masks


def compute_pod_far_by_regions_from_directory(
    nc_dir: str | Path,
    ref_name: str,
    *,
    shp_path: str | Path,
    region_field: str,
    group: str = "all",  # "all" | "season" | "month"
    pattern: str = "*.TIMEFIX.daily.CHINA.nc",
    precip_var: str | None = None,
    threshold: float = 1.0,
    time_range: Tuple[str, str] | None = None,
    seasons: Mapping[str, Sequence[int]] | None = None,
    months: Sequence[int] | None = None,
) -> Tuple[object, List[str], List[str]]:
    """
    按分区（shp + 字段）计算 POD/FAR，并可按季节/月份分组。

    返回
    ----
    stats, products, region_names

    - group="all":
        stats: dict[region][product] -> (POD, FAR)
    - group="season":
        stats: dict[region][season][product] -> (POD, FAR)
    - group="month":
        stats: dict[region][month][product] -> (POD, FAR)
    """
    from .evaluation import detect_precip_var, compute_pod_far_xr  # 若同文件可删

    nc_dir = Path(nc_dir)
    paths = sorted(nc_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"目录 {nc_dir} 中未找到 {pattern} 文件。")

    ref_path = None
    product_paths: Dict[str, Path] = {}
    for p in paths:
        prod = p.name.split(".")[0]
        if prod == ref_name:
            ref_path = p
        else:
            product_paths[prod] = p

    if ref_path is None:
        raise FileNotFoundError(f"未找到参考产品 {ref_name} 对应的文件（匹配 {pattern}）")

    # 参考数据
    ref_ds = xr.open_dataset(ref_path, chunks={"time": 90})
    ref_var = precip_var or detect_precip_var(ref_ds)
    ref_da = ref_ds[ref_var]
    if time_range is not None:
        ref_da = ref_da.sel(time=slice(time_range[0], time_range[1]))

    # 分区掩膜（基于参考网格）
    region_names, region_masks = build_region_masks_from_shp(ref_da, shp_path, region_field)

    # 分组定义
    if seasons is None:
        seasons = {
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Autumn": [9, 10, 11],
            "Winter": [12, 1, 2],
        }
    if months is None:
        months = list(range(1, 13))

    def _subset_by_months(da: xr.DataArray, mon_list: Sequence[int]) -> xr.DataArray:
        return da.sel(time=da["time"].dt.month.isin(mon_list))

    products = list(product_paths.keys())

    if group == "all":
        stats: Dict[str, Dict[str, Tuple[float, float]]] = {r: {} for r in region_names}

        for prod, path in product_paths.items():
            sim_ds = xr.open_dataset(path, chunks={"time": 90})
            sim_var = precip_var or detect_precip_var(sim_ds)
            sim_da = sim_ds[sim_var]
            if time_range is not None:
                sim_da = sim_da.sel(time=slice(time_range[0], time_range[1]))

            for r in region_names:
                pod, far = compute_pod_far_xr(ref_da, sim_da, threshold, region_masks[r])
                stats[r][prod] = (pod, far)

            sim_ds.close()

        ref_ds.close()
        return stats, products, region_names

    elif group == "season":
        stats2: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {
            r: {s: {} for s in seasons.keys()} for r in region_names
        }

        for prod, path in product_paths.items():
            sim_ds = xr.open_dataset(path, chunks={"time": 90})
            sim_var = precip_var or detect_precip_var(sim_ds)
            sim_da = sim_ds[sim_var]
            if time_range is not None:
                sim_da = sim_da.sel(time=slice(time_range[0], time_range[1]))

            for r in region_names:
                msk = region_masks[r]
                for sname, mon_list in seasons.items():
                    obs_s = _subset_by_months(ref_da, mon_list)
                    sim_s = _subset_by_months(sim_da, mon_list)
                    pod, far = compute_pod_far_xr(obs_s, sim_s, threshold, msk)
                    stats2[r][sname][prod] = (pod, far)

            sim_ds.close()

        ref_ds.close()
        return stats2, products, region_names

    elif group == "month":
        stats3: Dict[str, Dict[int, Dict[str, Tuple[float, float]]]] = {
            r: {m: {} for m in months} for r in region_names
        }

        for prod, path in product_paths.items():
            sim_ds = xr.open_dataset(path, chunks={"time": 90})
            sim_var = precip_var or detect_precip_var(sim_ds)
            sim_da = sim_ds[sim_var]
            if time_range is not None:
                sim_da = sim_da.sel(time=slice(time_range[0], time_range[1]))

            for r in region_names:
                msk = region_masks[r]
                for m in months:
                    obs_m = _subset_by_months(ref_da, [m])
                    sim_m = _subset_by_months(sim_da, [m])
                    pod, far = compute_pod_far_xr(obs_m, sim_m, threshold, msk)
                    stats3[r][m][prod] = (pod, far)

            sim_ds.close()

        ref_ds.close()
        return stats3, products, region_names

    else:
        ref_ds.close()
        raise ValueError("group 仅支持 'all' | 'season' | 'month'")


