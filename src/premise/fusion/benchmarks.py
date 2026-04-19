# -*- coding: utf-8 -*-
"""
premise.fusion.benchmarks
========================

Benchmark / ablation fusion suite for precipitation products.

Methods supported (from config["benchmark"]["methods"]):
- evw_ref_zone_month
- evw_tc_zone_month
- bma_gaussian_zone_month
- qm_blend_mean_zone_month
- geo_residual_idw_month
- ml_ridge_zone
- ml_rf_zone
- ml_gbdt_zone

Public APIs:
- fit_params(config)                   -> bench_params.joblib
- fit_ml(config)                       -> ml_models.joblib
- fit_geo_residual_month(config, ...)  -> geo_residual_idw_month.joblib
- predict(config)                      -> fused NetCDF blocks per method
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
class FusionBenchmarksConfig:
    cfg: dict

    @staticmethod
    def from_json(path: PathLike) -> "FusionBenchmarksConfig":
        from ._common import load_json
        return FusionBenchmarksConfig(cfg=load_json(path))


def _pick_blocks(t_all: pd.DatetimeIndex, block_days: int):
    blocks = []
    cur = t_all.min()
    while cur <= t_all.max():
        end = min(cur + pd.Timedelta(days=block_days - 1), t_all.max())
        blocks.append((cur, end))
        cur = end + pd.Timedelta(days=1)
    return blocks


def sample_zone_month(
    ref: xr.DataArray,
    prods,
    climate_id: xr.DataArray,
    cid: int,
    mon: int,
    times: pd.DatetimeIndex,
    n_samp: int,
    min_n: int,
    rng: np.random.Generator,
):
    clim = climate_id.values
    valid = np.isfinite(clim)
    yy_all, xx_all = np.where(valid)
    cids_all = clim[yy_all, xx_all].astype(int)

    t_mon = times[times.month == mon]
    if len(t_mon) == 0:
        return None

    idx_cells = np.where(cids_all == int(cid))[0]
    if idx_cells.size == 0:
        return None

    cell_sel = rng.choice(idx_cells.size, size=n_samp, replace=True)
    time_sel = rng.integers(0, len(t_mon), size=n_samp, endpoint=False)

    y_idx = yy_all[idx_cells[cell_sel]]
    x_idx = xx_all[idx_cells[cell_sel]]

    t_lookup = pd.Index(ref["time"].values)
    t_idx = t_lookup.get_indexer(t_mon.values[time_sel])
    ok = t_idx >= 0
    t_idx, y_idx, x_idx = t_idx[ok], y_idx[ok], x_idx[ok]
    if t_idx.size < min_n:
        return None

    y = ref.isel(
        time=xr.DataArray(t_idx, dims="s"),
        lat=xr.DataArray(y_idx, dims="s"),
        lon=xr.DataArray(x_idx, dims="s"),
    ).values.astype("float32")

    X = np.column_stack([
        da.isel(
            time=xr.DataArray(t_idx, dims="s"),
            lat=xr.DataArray(y_idx, dims="s"),
            lon=xr.DataArray(x_idx, dims="s"),
        ).values.astype("float32")
        for _, da in prods
    ])

    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < min_n:
        return None
    return X[m], y[m]


def fit_params(config: Union[FusionBenchmarksConfig, dict, str, PathLike]) -> str:
    """
    Fit parameters for EVW/BMA/QM by zone-month.
    Returns path to bench_params.joblib.
    """
    require_optional(["joblib"], extra_hint="fusion")
    import joblib

    cfg = config.cfg if isinstance(config, FusionBenchmarksConfig) else as_config(config)
    b = cfg["benchmark"]

    ref, prods, climate_id, mapping = load_all_basic(cfg)

    train_rng = pd.date_range(b["train_time_range"][0], b["train_time_range"][1], freq="D")
    train_times = pd.DatetimeIndex(ref["time"].values).intersection(train_rng)
    if len(train_times) == 0:
        raise ValueError("benchmark.train_time_range has empty intersection with data time axis.")

    rng = np.random.default_rng(int(b.get("random_seed", 2025)))
    n_samp = int(b.get("samples_per_zone_month", 80000))
    min_n = int(b.get("min_samples_per_zone_month", 8000))
    eps = float(b.get("eps", 1e-6))
    qn = int(b.get("qm_n_quantiles", 401))
    qs = np.linspace(0, 1, qn, dtype="float32")

    prod_names = [n for n, _ in prods]
    K = len(prods)

    params: Dict[str, dict] = {
        "mapping": mapping,
        "product_names": prod_names,
        "evw_ref_zone_month": {},
        "evw_tc_zone_month": {},
        "bma_gaussian_zone_month": {},
        "qm_maps_zone_month": {},
    }

    for cid in sorted(mapping.keys()):
        for mon in range(1, 13):
            pack = sample_zone_month(ref, prods, climate_id, cid, mon, train_times, n_samp, min_n, rng)
            if pack is None:
                continue
            X, y = pack  # X: [N,K], y: [N]

            # EVW-REF
            mse = np.mean((X - y.reshape(-1, 1)) ** 2, axis=0).astype("float32")
            w_ref = 1.0 / (mse + eps)
            w_ref = (w_ref / w_ref.sum()).astype("float32")
            params["evw_ref_zone_month"][f"{cid:02d}_{mon:02d}"] = w_ref

            # EVW-TC: only first 3 products
            if K >= 3:
                C = np.cov(X[:, :3].T.astype("float64"), bias=True)
                v = np.diag(C)
                c12, c13, c23 = C[0, 1], C[0, 2], C[1, 2]

                def _tc_var(v1, c12_, c13_, c23_):
                    if np.abs(c23_) < 1e-12:
                        return np.nan
                    return v1 - (c12_ * c13_ / c23_)

                ve = np.array(
                    [
                        _tc_var(v[0], c12, c13, c23),
                        _tc_var(v[1], c12, c23, c13),
                        _tc_var(v[2], c13, c23, c12),
                    ],
                    dtype="float64",
                )
                if np.all(np.isfinite(ve)):
                    ve = np.maximum(ve, eps).astype("float32")
                    w_tc = (1.0 / ve)
                    w_tc = (w_tc / w_tc.sum()).astype("float32")
                    params["evw_tc_zone_month"][f"{cid:02d}_{mon:02d}"] = w_tc

            # BMA via EM on Gaussian errors
            res = y.reshape(-1, 1) - X
            sig2 = np.maximum(np.nanvar(res, axis=0), eps).astype("float64")
            w = np.ones(K, dtype="float64") / K
            for _ in range(30):
                ll = np.stack(
                    [
                        -0.5 * np.log(sig2[k]) - 0.5 * ((y - X[:, k]) ** 2) / sig2[k] + np.log(w[k] + eps)
                        for k in range(K)
                    ],
                    axis=1,
                )
                ll = ll - ll.max(axis=1, keepdims=True)
                r = np.exp(ll)
                r = r / (r.sum(axis=1, keepdims=True) + eps)
                w_new = r.mean(axis=0)
                sig2_new = (r * (y.reshape(-1, 1) - X) ** 2).sum(axis=0) / (r.sum(axis=0) + eps)
                if np.max(np.abs(w_new - w)) < 1e-6:
                    w, sig2 = w_new, sig2_new
                    break
                w, sig2 = w_new, sig2_new
            w = (w / w.sum()).astype("float32")
            params["bma_gaussian_zone_month"][f"{cid:02d}_{mon:02d}"] = {"w": w, "sig2": sig2.astype("float32")}

            # QM maps: pooled by zone-month
            xq = np.vstack([np.quantile(X[:, k], qs) for k in range(K)]).astype("float32")  # [K,qn]
            yq = np.quantile(y, qs).astype("float32")  # [qn]
            params["qm_maps_zone_month"][f"{cid:02d}_{mon:02d}"] = {"xq": xq, "yq": yq}

    out_dir = ensure_dir(Path(b["write"]["out_dir"]).expanduser().resolve() / "models")
    ppath = out_dir / "bench_params.joblib"
    joblib.dump(params, ppath)
    logger.info("Saved benchmark params: %s", str(ppath))
    return str(ppath)


def idw_grid(xy_s, v_s, xy_g, power=2.0, eps=1e-12):
    dx = xy_g[:, None, 0] - xy_s[None, :, 0]
    dy = xy_g[:, None, 1] - xy_s[None, :, 1]
    d2 = dx * dx + dy * dy + eps
    w = 1.0 / (d2 ** (power / 2.0))
    wsum = w.sum(axis=1)
    return (w @ v_s) / wsum


def fit_geo_residual_month(
    config: Union[FusionBenchmarksConfig, dict, str, PathLike],
    base_method: str = "evw_ref_zone_month",
) -> str:
    """
    Fit monthly residual surfaces via IDW (geostatistical adjustment) based on a base method.
    Returns path to geo_residual_idw_month.joblib.
    """
    require_optional(["joblib"], extra_hint="fusion")
    import joblib

    cfg = config.cfg if isinstance(config, FusionBenchmarksConfig) else as_config(config)
    b = cfg["benchmark"]

    ref, prods, climate_id, mapping = load_all_basic(cfg)

    out_dir = Path(b["write"]["out_dir"]).expanduser().resolve()
    ensure_dir(out_dir / "models")

    ppath = out_dir / "models" / "bench_params.joblib"
    if not ppath.exists():
        raise FileNotFoundError("Run fit_params first to create bench_params.joblib.")
    params = joblib.load(ppath)

    eps = float(b.get("eps", 1e-6))
    geo_cfg = b.get("geo", {}) or {}
    npts = int(geo_cfg.get("n_points_per_month", 6000))
    power = float(geo_cfg.get("power_idw", 2.0))
    rng = np.random.default_rng(int(b.get("random_seed", 2025)))

    train_rng = pd.date_range(b["train_time_range"][0], b["train_time_range"][1], freq="D")
    train_times = pd.DatetimeIndex(ref["time"].values).intersection(train_rng)

    lat = ref["lat"].values
    lon = ref["lon"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    grid_xy = np.column_stack([lon2d.ravel(), lat2d.ravel()]).astype("float32")

    clim = climate_id.values
    valid = np.isfinite(clim).ravel()

    base_month = {}
    for mon in range(1, 13):
        t_mon = train_times[train_times.month == mon]
        if len(t_mon) == 0:
            continue

        Y = ref.sel(time=t_mon).mean("time", skipna=True).values.astype("float32")
        X = np.stack([da.sel(time=t_mon).mean("time", skipna=True).values.astype("float32") for _, da in prods], axis=0)  # [K,lat,lon]
        K = X.shape[0]

        # weights by zone-month
        W = np.full((len(lat), len(lon), K), np.nan, dtype="float32")
        for cid in np.unique(clim[np.isfinite(clim)].astype(int)):
            key = f"{cid:02d}_{mon:02d}"
            if key not in params.get(base_method, {}):
                continue
            w = params[base_method][key].astype("float32")
            m = (clim.astype(int) == cid)
            for k in range(min(K, len(w))):
                W[..., k][m] = w[k]

        Pbase = np.nansum(X * np.moveaxis(W, -1, 0), axis=0)  # [lat,lon]
        r = (Y - Pbase).astype("float32")

        r_flat = r.ravel()
        idx_valid = np.where(valid & np.isfinite(r_flat))[0]
        if idx_valid.size == 0:
            continue
        sel = rng.choice(idx_valid, size=min(npts, idx_valid.size), replace=False)
        xy_s = grid_xy[sel]
        v_s = r_flat[sel]

        v_g = np.full((grid_xy.shape[0],), np.nan, dtype="float32")
        v_g[valid] = idw_grid(xy_s, v_s, grid_xy[valid], power=power, eps=eps).astype("float32")
        base_month[mon] = v_g.reshape(len(lat), len(lon))

    gpath = out_dir / "models" / "geo_residual_idw_month.joblib"
    joblib.dump({"base_method": base_method, "residual_idw_month": base_month}, gpath)
    logger.info("Saved GEO residual surfaces: %s", str(gpath))
    return str(gpath)


def _get_w_for_method(params, method, key, K, eps=1e-6, fallback_equal=True):
    if method == "evw_ref_zone_month":
        if key in params.get("evw_ref_zone_month", {}):
            return params["evw_ref_zone_month"][key].astype("float32")
    elif method == "evw_tc_zone_month":
        if key in params.get("evw_tc_zone_month", {}):
            w = params["evw_tc_zone_month"][key].astype("float32")
            if len(w) < K:
                w2 = np.zeros((K,), dtype="float32")
                w2[: len(w)] = w
                w = w2
            return w
    elif method == "bma_gaussian_zone_month":
        if key in params.get("bma_gaussian_zone_month", {}):
            return params["bma_gaussian_zone_month"][key]["w"].astype("float32")
    else:
        return None

    if fallback_equal:
        return (np.ones(K, dtype="float32") / K)
    return None


def blend_nanaware(sub, w, eps=1e-6):
    w3 = w.astype("float32").reshape(1, 1, -1)
    finite = np.isfinite(sub)
    den = (w3 * finite).sum(axis=2)
    num = np.nansum(sub * w3, axis=2)
    out = num / (den + eps)
    out[den <= eps] = np.nan
    return out.astype("float32")


def fit_ml(config: Union[FusionBenchmarksConfig, dict, str, PathLike]) -> str:
    """
    Train zone-wise ML models (Ridge/RF/GBDT). Returns path to ml_models.joblib.
    """
    require_optional(["joblib", "sklearn"], extra_hint="fusion")
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    cfg = config.cfg if isinstance(config, FusionBenchmarksConfig) else as_config(config)
    b = cfg["benchmark"]
    ml = b.get("ml", {}) or {}

    ref, prods, climate_id, mapping = load_all_basic(cfg)

    train_rng = pd.date_range(b["train_time_range"][0], b["train_time_range"][1], freq="D")
    train_times = pd.DatetimeIndex(ref["time"].values).intersection(train_rng)
    if len(train_times) == 0:
        raise ValueError("benchmark.train_time_range has empty intersection with data time axis.")

    rng = np.random.default_rng(int(b.get("random_seed", 2025)))
    n_zone = int(ml.get("n_samples_per_zone", 300000))

    clim = climate_id.values
    valid = np.isfinite(clim)
    yy_all, xx_all = np.where(valid)
    cids_all = clim[yy_all, xx_all].astype(int)

    t_lookup = pd.Index(ref["time"].values)
    t_idx_all = t_lookup.get_indexer(train_times.values)
    t_idx_all = t_idx_all[t_idx_all >= 0]

    out_dir = Path(b["write"]["out_dir"]).expanduser().resolve()
    ensure_dir(out_dir / "models")

    models = {"ml_ridge_zone": {}, "ml_rf_zone": {}, "ml_gbdt_zone": {}}

    for cid in sorted(mapping.keys()):
        idx_cells = np.where(cids_all == int(cid))[0]
        if idx_cells.size == 0:
            continue

        s = n_zone
        cell_sel = rng.choice(idx_cells.size, size=s, replace=True)
        time_sel = rng.integers(0, len(t_idx_all), size=s, endpoint=False)

        y_idx = yy_all[idx_cells[cell_sel]]
        x_idx = xx_all[idx_cells[cell_sel]]
        t_idx = t_idx_all[time_sel]

        y = ref.isel(
            time=xr.DataArray(t_idx, dims="s"),
            lat=xr.DataArray(y_idx, dims="s"),
            lon=xr.DataArray(x_idx, dims="s"),
        ).values.astype("float32")

        X = np.column_stack([
            da.isel(
                time=xr.DataArray(t_idx, dims="s"),
                lat=xr.DataArray(y_idx, dims="s"),
                lon=xr.DataArray(x_idx, dims="s"),
            ).values.astype("float32")
            for _, da in prods
        ])

        m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if m.sum() < max(20000, int(0.1 * n_zone)):
            continue
        X, y = X[m].astype("float32"), y[m].astype("float32")

        ridge = Ridge(alpha=float(ml.get("ridge", {}).get("alpha", 1.0)))
        ridge.fit(X, y)
        models["ml_ridge_zone"][str(cid)] = ridge

        rf_cfg = ml.get("rf", {}) or {}
        rf = RandomForestRegressor(
            n_estimators=int(rf_cfg.get("n_estimators", 300)),
            max_depth=None if rf_cfg.get("max_depth", None) is None else int(rf_cfg["max_depth"]),
            n_jobs=int(rf_cfg.get("n_jobs", 1)),
            random_state=int(rf_cfg.get("random_seed", 2025)),
        )
        rf.fit(X, y)
        models["ml_rf_zone"][str(cid)] = rf

        g_cfg = ml.get("gbdt", {}) or {}
        gbdt = GradientBoostingRegressor(
            n_estimators=int(g_cfg.get("n_estimators", 400)),
            learning_rate=float(g_cfg.get("learning_rate", 0.05)),
            max_depth=int(g_cfg.get("max_depth", 3)),
            random_state=int(g_cfg.get("random_seed", 2025)),
        )
        gbdt.fit(X, y)
        models["ml_gbdt_zone"][str(cid)] = gbdt

    mpath = out_dir / "models" / "ml_models.joblib"
    joblib.dump(models, mpath)
    logger.info("Saved ML models: %s", str(mpath))
    return str(mpath)


def predict(config: Union[FusionBenchmarksConfig, dict, str, PathLike]) -> Dict[str, str]:
    """
    Predict fused precipitation for each method listed in cfg["benchmark"]["methods"].
    Returns dict: {method: out_dir}
    """
    require_optional(["joblib"], extra_hint="fusion")
    import joblib

    cfg = config.cfg if isinstance(config, FusionBenchmarksConfig) else as_config(config)
    b = cfg["benchmark"]
    pp = cfg.get("preprocess", {}) or {}
    clip_nonneg = bool(pp.get("clip_nonneg", True))
    eps = float(b.get("eps", 1e-6))
    fallback_equal = bool(b.get("fallback_equal_weight", True))

    ref, prods, climate_id, mapping = load_all_basic(cfg)

    apply_rng = pd.date_range(b["apply_time_range"][0], b["apply_time_range"][1], freq="D")
    times = pd.DatetimeIndex(ref["time"].values).intersection(apply_rng)
    if len(times) == 0:
        raise ValueError("benchmark.apply_time_range has empty intersection with data time axis.")

    out_dir = Path(b["write"]["out_dir"]).expanduser().resolve()
    ensure_dir(out_dir)

    params_path = out_dir / "models" / "bench_params.joblib"
    if not params_path.exists():
        raise FileNotFoundError("bench_params.joblib not found. Run fit_params first.")
    params = joblib.load(params_path)

    geo_path = out_dir / "models" / "geo_residual_idw_month.joblib"
    geo_obj = joblib.load(geo_path) if geo_path.exists() else None

    methods = list(b["methods"])
    block_days = int(b["write"].get("time_block_days", 31))
    out_var = b["write"].get("out_var", "pr_fused")

    # stack valid indices
    base_stack = ref.isel(time=0).stack(z=("lat", "lon"))
    clim_stack = climate_id.stack(z=("lat", "lon"))
    valid = np.isfinite(clim_stack.values)
    flat_idx = np.where(valid)[0]
    z_coord = base_stack["z"].isel(z=flat_idx)
    cid_flat = clim_stack.isel(z=flat_idx).values.astype(int)
    uniq_cids = np.unique(cid_flat)

    K = len(prods)
    blocks = _pick_blocks(times, block_days)
    results: Dict[str, str] = {}

    # load ML if needed
    ml_obj = None
    if any(m.startswith("ml_") for m in methods):
        mdl_path = out_dir / "models" / "ml_models.joblib"
        if not mdl_path.exists():
            raise FileNotFoundError("ml_models.joblib not found. Run fit_ml first.")
        ml_obj = joblib.load(mdl_path)

    for method in methods:
        mdir = ensure_dir(out_dir / method)
        results[method] = str(mdir)

        for t0, t1 in blocks:
            tsel = times[(times >= t0) & (times <= t1)]
            if len(tsel) == 0:
                continue

            # products -> P (T,Z,K)
            P_list = []
            for _, da in prods:
                da_blk = da.sel(time=tsel).stack(z=("lat", "lon")).isel(z=flat_idx)
                arr = da_blk.values.astype("float32")
                if clip_nonneg:
                    arr = np.maximum(arr, 0.0)
                P_list.append(arr)
            P = np.stack(P_list, axis=2)  # (T,Z,K)

            T, Z, _ = P.shape
            fused = np.full((T, Z), np.nan, dtype="float32")
            months = np.array([pd.Timestamp(t).month for t in tsel], dtype=int)

            if method in ("evw_ref_zone_month", "evw_tc_zone_month", "bma_gaussian_zone_month"):
                for mon in np.unique(months):
                    tidx = np.where(months == mon)[0]
                    for cid in uniq_cids:
                        zidx = np.where(cid_flat == cid)[0]
                        if zidx.size == 0:
                            continue
                        key = f"{cid:02d}_{mon:02d}"
                        w = _get_w_for_method(params, method, key, K, eps=eps, fallback_equal=fallback_equal)
                        if w is None:
                            continue
                        sub = P[np.ix_(tidx, zidx)]  # (Tsel,Zsel,K)
                        fused[np.ix_(tidx, zidx)] = blend_nanaware(sub, w, eps=eps)

            elif method == "qm_blend_mean_zone_month":
                for mon in np.unique(months):
                    tidx = np.where(months == mon)[0]
                    for cid in uniq_cids:
                        zidx = np.where(cid_flat == cid)[0]
                        if zidx.size == 0:
                            continue
                        key = f"{cid:02d}_{mon:02d}"
                        if key not in params.get("qm_maps_zone_month", {}):
                            if fallback_equal:
                                fused[np.ix_(tidx, zidx)] = np.nanmean(P[np.ix_(tidx, zidx)], axis=2)
                            continue
                        mp = params["qm_maps_zone_month"][key]
                        xq = mp["xq"]  # (K,qn)
                        yq = mp["yq"]  # (qn,)
                        sub = P[np.ix_(tidx, zidx)]  # (Tsel,Zsel,K)
                        Q = np.empty_like(sub)
                        for k in range(K):
                            x = sub[:, :, k]
                            Q[:, :, k] = np.interp(x.ravel(), xq[k], yq).reshape(x.shape)
                        fused[np.ix_(tidx, zidx)] = np.nanmean(Q, axis=2)

            elif method == "geo_residual_idw_month":
                for mon in np.unique(months):
                    tidx = np.where(months == mon)[0]
                    rs = None
                    if geo_obj is not None and "residual_idw_month" in geo_obj and mon in geo_obj["residual_idw_month"]:
                        rs2d = geo_obj["residual_idw_month"][mon]
                        rs = xr.DataArray(rs2d, coords={"lat": ref["lat"].values, "lon": ref["lon"].values}, dims=("lat", "lon")) \
                                .stack(z=("lat", "lon")).isel(z=flat_idx).values.astype("float32")

                    for cid in uniq_cids:
                        zidx = np.where(cid_flat == cid)[0]
                        if zidx.size == 0:
                            continue
                        key = f"{cid:02d}_{mon:02d}"
                        w = _get_w_for_method(params, "evw_ref_zone_month", key, K, eps=eps, fallback_equal=fallback_equal)
                        if w is None:
                            continue
                        sub = P[np.ix_(tidx, zidx)]
                        base = blend_nanaware(sub, w, eps=eps)
                        if rs is not None:
                            base = base + rs[zidx].reshape(1, -1)
                        fused[np.ix_(tidx, zidx)] = base

            elif method.startswith("ml_"):
                for cid in uniq_cids:
                    zidx = np.where(cid_flat == cid)[0]
                    if zidx.size == 0:
                        continue
                    mdl = ml_obj[method].get(str(cid), None) if ml_obj is not None else None
                    sub = P[:, zidx, :]  # (T,Zsel,K)
                    X = sub.reshape(-1, K).astype("float32")
                    base = np.nanmean(X, axis=1).astype("float32")
                    base = np.where(np.isfinite(base), base, 0.0).astype("float32")

                    yhat = base.copy()
                    if mdl is not None:
                        good = np.isfinite(X).all(axis=1)
                        if np.any(good):
                            yhat[good] = mdl.predict(X[good]).astype("float32")
                    fused[:, zidx] = yhat.reshape(T, zidx.size)

            else:
                raise ValueError(f"Unknown benchmark method: {method}")

            if clip_nonneg:
                fused = np.maximum(fused, 0.0)

            da_out = xr.DataArray(
                fused,
                coords={"time": tsel.values, "z": z_coord},
                dims=("time", "z"),
                name=out_var,
            ).unstack("z")

            ds_out = da_out.to_dataset()
            fn = mdir / f"fused_{tsel[0].strftime('%Y%m%d')}_{tsel[-1].strftime('%Y%m%d')}.nc"
            encoding = {out_var: {"zlib": True, "complevel": 4}}
            ds_out.to_netcdf(fn, encoding=encoding)

    return results
