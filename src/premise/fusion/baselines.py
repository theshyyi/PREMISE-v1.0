# -*- coding: utf-8 -*-
"""
premise.fusion.baselines
=======================

Baseline fusion methods for corrected precipitation products.

Config schema (example):
{
  "io": {...},
  "preprocess": {...},
  "baseline": {
    "methods": ["mean","median","inv_rmse_zone_month","stacking_pos_zone_month","best_single_zone_month"],
    "train_time_range": ["2000-01-01","2015-12-31"],
    "apply_time_range": ["2000-01-01","2022-12-31"],
    "out_dir": "/path/to/out/baselines",
    "out_var": "pr_fused",
    "samples_per_zone_month": 60000,
    "min_samples_per_zone_month": 5000,
    "random_seed": 2025,
    "eps": 1e-6,
    "write": {"time_block_days": 31, "lat_block": 120}
  }
}

Public APIs:
- fit_baseline_weights(config)
- fuse_baselines(config, method=None)  # method None -> run all methods in config
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._common import PathLike, as_config, ensure_dir, load_all_basic, require_optional

logger = logging.getLogger(__name__)


@dataclass
class FusionBaselinesConfig:
    cfg: dict

    @staticmethod
    def from_json(path: PathLike) -> "FusionBaselinesConfig":
        from ._common import load_json
        return FusionBaselinesConfig(cfg=load_json(path))


def _safe_rmse(sim: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((sim[m] - obs[m]) ** 2)))


def _pick_blocks(t_all: pd.DatetimeIndex, block_days: int):
    blocks = []
    cur = t_all.min()
    while cur <= t_all.max():
        end = min(cur + pd.Timedelta(days=block_days - 1), t_all.max())
        blocks.append((cur, end))
        cur = end + pd.Timedelta(days=1)
    return blocks


def _vextract_3d(da: xr.DataArray, t_idx, y_idx, x_idx) -> np.ndarray:
    return da.isel(
        time=xr.DataArray(t_idx, dims="sample"),
        lat=xr.DataArray(y_idx, dims="sample"),
        lon=xr.DataArray(x_idx, dims="sample"),
    ).values.astype("float32")


def _blend_nanaware(subK: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    subK: (K, N)
    w: (K,)
    """
    w = w.astype("float32").reshape(-1, 1)
    finite = np.isfinite(subK)
    num = np.nansum(subK * w, axis=0)
    den = np.sum(w * finite, axis=0)
    out = num / (den + eps)
    out[den <= eps] = np.nan
    return out.astype("float32")


def fit_baseline_weights(config: Union[FusionBaselinesConfig, dict, str, PathLike]) -> str:
    """
    Fit zone-month weights needed by:
      inv_rmse_zone_month, stacking_pos_zone_month, best_single_zone_month

    Returns path to joblib weights file.
    """
    require_optional(["joblib", "tqdm", "sklearn"], extra_hint="fusion")
    import joblib
    from tqdm import tqdm
    from sklearn.linear_model import LinearRegression

    cfg = config.cfg if isinstance(config, FusionBaselinesConfig) else as_config(config)
    bcfg = cfg["baseline"]

    out_dir = Path(bcfg["out_dir"]).expanduser().resolve()
    wdir = ensure_dir(out_dir / "weights")

    ref, prods, climate_id, mapping = load_all_basic(cfg)

    train_rng = pd.date_range(bcfg["train_time_range"][0], bcfg["train_time_range"][1], freq="D")
    train_times = pd.DatetimeIndex(ref["time"].values).intersection(train_rng)
    if len(train_times) == 0:
        raise ValueError("baseline.train_time_range has empty intersection with data time axis.")

    clim = climate_id.values.astype("float32")
    valid_cells = np.isfinite(clim)
    yy_all, xx_all = np.where(valid_cells)
    cids_all = clim[yy_all, xx_all].astype(int)

    prod_names = [n for n, _ in prods]
    K = len(prods)

    rng = np.random.default_rng(int(bcfg.get("random_seed", 2025)))
    n_samp = int(bcfg.get("samples_per_zone_month", 60000))
    min_n = int(bcfg.get("min_samples_per_zone_month", 5000))
    eps = float(bcfg.get("eps", 1e-6))

    weights: Dict[str, dict] = {
        "mapping": mapping,
        "product_names": prod_names,
        "inv_rmse_zone_month": {},
        "stacking_pos_zone_month": {},
        "best_single_zone_month": {},
    }

    t_lookup = pd.Index(ref["time"].values)
    for cid in tqdm(sorted(mapping.keys()), desc="Fit baseline weights (zones)"):
        for mon in range(1, 13):
            t_mon = train_times[train_times.month == mon]
            if len(t_mon) == 0:
                continue

            idx_cells = np.where(cids_all == int(cid))[0]
            if idx_cells.size == 0:
                continue

            cell_sel = rng.choice(idx_cells.size, size=n_samp, replace=True)
            time_sel = rng.integers(0, len(t_mon), size=n_samp, endpoint=False)

            y_idx = yy_all[idx_cells[cell_sel]]
            x_idx = xx_all[idx_cells[cell_sel]]

            t_idx = t_lookup.get_indexer(t_mon.values[time_sel])
            ok = t_idx >= 0
            t_idx, y_idx, x_idx = t_idx[ok], y_idx[ok], x_idx[ok]
            if t_idx.size < min_n:
                continue

            y = _vextract_3d(ref, t_idx, y_idx, x_idx)
            P = np.column_stack([_vextract_3d(da, t_idx, y_idx, x_idx) for _, da in prods])

            m = np.isfinite(y) & np.all(np.isfinite(P), axis=1)
            if m.sum() < min_n:
                continue
            y = y[m].astype("float32")
            P = P[m].astype("float32")

            rmses = np.array([_safe_rmse(P[:, k], y) for k in range(K)], dtype="float32")
            if not np.all(np.isfinite(rmses)):
                continue
            w = 1.0 / (rmses + eps) ** 2
            w = (w / np.sum(w)).astype("float32")
            weights["inv_rmse_zone_month"][f"{cid:02d}_{mon:02d}"] = w

            best_k = int(np.argmin(rmses))
            weights["best_single_zone_month"][f"{cid:02d}_{mon:02d}"] = best_k

            lr = LinearRegression(positive=True, fit_intercept=False)
            lr.fit(P, y)
            w2 = np.maximum(lr.coef_.astype("float32"), 0.0)
            if w2.sum() <= 0:
                w2 = w.copy()
            else:
                w2 = (w2 / w2.sum()).astype("float32")
            weights["stacking_pos_zone_month"][f"{cid:02d}_{mon:02d}"] = w2

    wpath = wdir / "baseline_weights.joblib"
    joblib.dump(weights, wpath)
    logger.info("Saved baseline weights: %s", str(wpath))
    return str(wpath)


def fuse_baselines(config: Union[FusionBaselinesConfig, dict, str, PathLike], method: Optional[str] = None) -> Dict[str, str]:
    """
    Apply baseline fusion methods (block-wise). If method is None, runs all methods in cfg["baseline"]["methods"].
    Returns dict: {method: output_dir}
    """
    require_optional(["joblib", "tqdm", "sklearn"], extra_hint="fusion")
    import joblib
    from tqdm import tqdm

    cfg = config.cfg if isinstance(config, FusionBaselinesConfig) else as_config(config)
    bcfg = cfg["baseline"]
    methods = [method] if method else list(bcfg["methods"])

    out_dir = Path(bcfg["out_dir"]).expanduser().resolve()
    ensure_dir(out_dir)

    ref, prods, climate_id, mapping = load_all_basic(cfg)

    apply_rng = pd.date_range(bcfg["apply_time_range"][0], bcfg["apply_time_range"][1], freq="D")
    all_times = pd.DatetimeIndex(ref["time"].values).intersection(apply_rng)
    if len(all_times) == 0:
        raise ValueError("baseline.apply_time_range has empty intersection with data time axis.")

    wobj = None
    if any("zone_month" in m for m in methods):
        wpath = out_dir / "weights" / "baseline_weights.joblib"
        if not wpath.exists():
            fit_baseline_weights(cfg)
        wobj = joblib.load(wpath)

    write_cfg = bcfg.get("write", {}) or {}
    block_days = int(write_cfg.get("time_block_days", 31))
    lat_block = int(write_cfg.get("lat_block", 120))
    out_var = bcfg.get("out_var", "pr_fused")
    clip_nonneg = bool(cfg.get("preprocess", {}).get("clip_nonneg", True))
    eps = float(bcfg.get("eps", 1e-6))

    lat = ref["lat"].values
    lon = ref["lon"].values
    nlat, nlon = len(lat), len(lon)
    clim = climate_id.values.astype("float32")

    blocks = _pick_blocks(all_times, block_days)
    results: Dict[str, str] = {}

    for mth in methods:
        out_root = ensure_dir(out_dir / mth)
        results[mth] = str(out_root)

        for b0, b1 in tqdm(blocks, desc=f"Baseline fusion: {mth}"):
            tsel = all_times[(all_times >= b0) & (all_times <= b1)]
            if len(tsel) == 0:
                continue

            out = np.full((len(tsel), nlat, nlon), np.nan, dtype="float32")

            for y0 in range(0, nlat, lat_block):
                y1 = min(nlat, y0 + lat_block)
                clim_blk = clim[y0:y1, :]
                valid_blk = np.isfinite(clim_blk)
                if not np.any(valid_blk):
                    continue

                for it, tt in enumerate(tsel):
                    sub = np.stack(
                        [da.sel(time=tt).isel(lat=slice(y0, y1)).values.astype("float32") for _, da in prods],
                        axis=0,
                    )  # (K, yblk, nlon)

                    if clip_nonneg:
                        sub = np.maximum(sub, 0.0)

                    if mth == "mean":
                        fused_blk = np.nanmean(sub, axis=0)

                    elif mth == "median":
                        fused_blk = np.nanmedian(sub, axis=0)

                    elif mth in ("inv_rmse_zone_month", "stacking_pos_zone_month", "best_single_zone_month"):
                        mon = int(pd.Timestamp(tt).month)
                        fused_blk = np.full((y1 - y0, nlon), np.nan, dtype="float32")

                        # process per zone
                        for cid in np.unique(clim_blk[valid_blk].astype(int)):
                            key = f"{cid:02d}_{mon:02d}"
                            mask = (clim_blk.astype(int) == cid)

                            if mth == "best_single_zone_month":
                                if key not in wobj["best_single_zone_month"]:
                                    continue
                                k = int(wobj["best_single_zone_month"][key])
                                fused_blk[mask] = sub[k, :, :][mask]
                            else:
                                dict_key = "inv_rmse_zone_month" if mth == "inv_rmse_zone_month" else "stacking_pos_zone_month"
                                if key not in wobj[dict_key]:
                                    continue
                                w = wobj[dict_key][key].astype("float32")
                                # nan-aware weighted blend on masked pixels
                                sub_m = sub[:, mask]  # (K, N)
                                fused_blk[mask] = _blend_nanaware(sub_m, w, eps=eps)

                    else:
                        raise ValueError(f"Unknown baseline method: {mth}")

                    if clip_nonneg:
                        fused_blk = np.maximum(fused_blk, 0.0)
                    out[it, y0:y1, :] = fused_blk

            ds_out = xr.Dataset(
                {out_var: (("time", "lat", "lon"), out)},
                coords={"time": tsel.values, "lat": lat, "lon": lon},
            )
            fn = out_root / f"fused_{tsel[0].strftime('%Y%m%d')}_{tsel[-1].strftime('%Y%m%d')}.nc"
            encoding = {out_var: {"zlib": True, "complevel": 4}}
            ds_out.to_netcdf(fn, encoding=encoding)

    return results
