# -*- coding: utf-8 -*-
"""
premise.fusion.rf_twostage
=========================

Two-stage ML precipitation fusion (per climate zone) on a reference grid.

This module integrates the logic from `precip_fusion_rf_twostage.py` into PREMISE as
a reusable library component.

Inputs (daily, mm/day):
  - Multiple corrected precipitation products (e.g., 3)
  - Static covariates (2D): DEM, slope, aspect, hillshade, ...
  - Dynamic covariates (3D): temp, wind, rhum, etc.
  - Climate zone shapefile (polygons) to build a per-zone mask

Outputs:
  - A fused precipitation NetCDF on the reference grid (variable: pr_fused)
  - A trained model bundle (joblib) containing per-zone models + feature imputer medians

Config:
  Supports the v2 JSON schema you provided (keys: reference, precip_products,
  static_covariates, dynamic_covariates, climate_zones, io, compute, training, models, output).

Optional dependencies (recommended extras):
  - scikit-learn, geopandas, regionmask, shapely, joblib, tqdm
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


def _require_optional() -> None:
    missing = []
    try:
        import geopandas as _  # noqa: F401
    except Exception:
        missing.append("geopandas")
    try:
        import regionmask as _  # noqa: F401
    except Exception:
        missing.append("regionmask")
    try:
        import joblib as _  # noqa: F401
    except Exception:
        missing.append("joblib")
    try:
        from tqdm import tqdm as _  # noqa: F401
    except Exception:
        missing.append("tqdm")
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: F401
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier  # noqa: F401
    except Exception:
        missing.append("scikit-learn")

    if missing:
        raise ImportError(
            "Fusion RF module requires optional dependencies: "
            + ", ".join(missing)
            + ". Install e.g. `pip install premise[fusion]`."
        )


def ensure_latlon(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for k in list(ds.coords):
        lk = k.lower()
        if lk in ["latitude", "y"] and "lat" not in ds.coords:
            rename[k] = "lat"
        if lk in ["longitude", "x"] and "lon" not in ds.coords:
            rename[k] = "lon"
    if rename:
        ds = ds.rename(rename)
    return ds


def normalize_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize time axis to pandas datetime64[ns] at daily resolution (00:00:00).
    Compatible with cftime/object times. Only uses YYYY-MM-DD, ignoring hour.
    """
    if "time" not in ds.coords:
        return ds

    t = ds["time"].values
    t_str = [str(x)[:10] for x in t]
    t2 = pd.to_datetime(t_str, errors="coerce")

    if pd.isna(t2).any():
        bad_idx = np.where(pd.isna(t2))[0][:10]
        bad_vals = [t_str[i] for i in bad_idx]
        raise ValueError(f"Failed to parse some time values to datetime. Examples: {bad_vals}")

    t2 = pd.DatetimeIndex(t2).normalize()
    return ds.assign_coords(time=t2)


def open_ds(path: PathLike, chunks: Optional[Dict[str, int]] = None) -> xr.Dataset:
    ds = xr.open_dataset(path, chunks=chunks)
    ds = ensure_latlon(ds)
    ds = normalize_time(ds)
    return ds


def align_grid(src: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    if np.array_equal(src["lat"].values, target["lat"].values) and np.array_equal(src["lon"].values, target["lon"].values):
        return src
    return src.interp(lat=target["lat"], lon=target["lon"], method="linear")


def time_intersection(datasets: List[xr.Dataset]) -> Tuple[List[xr.Dataset], pd.DatetimeIndex]:
    times = None
    for ds in datasets:
        if "time" in ds.coords:
            t = pd.DatetimeIndex(ds["time"].values).normalize()
            times = t if times is None else times.intersection(t)

    if times is None or len(times) == 0:
        raise ValueError("time intersection is empty. Please check time axes.")

    times = pd.DatetimeIndex(times).sort_values()

    out = []
    for ds in datasets:
        if "time" in ds.coords:
            out.append(ds.sel(time=times))
        else:
            out.append(ds)
    return out, times


def cyclical_time_features(times: pd.DatetimeIndex):
    doy = times.dayofyear.values.astype(np.float32)
    mon = times.month.values.astype(np.float32)
    doy_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    doy_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    mon_sin = np.sin(2 * np.pi * mon / 12.0).astype(np.float32)
    mon_cos = np.cos(2 * np.pi * mon / 12.0).astype(np.float32)
    return doy_sin, doy_cos, mon_sin, mon_cos


def median_impute(X: np.ndarray, med: Optional[np.ndarray] = None):
    if med is None:
        med = np.nanmedian(X, axis=0)
    X2 = X.copy()
    inds = np.where(~np.isfinite(X2))
    if inds[0].size > 0:
        X2[inds] = np.take(med, inds[1])
    return X2, med


def build_climate_mask(shp_path: str, climate_field: str, ref: xr.Dataset):
    import geopandas as gpd
    import regionmask

    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError("Climate-zone shapefile has no CRS. Please define CRS.")
    # Ensure lon/lat in EPSG:4326
    if "epsg:4326" not in gdf.crs.to_string().lower() and "wgs84" not in gdf.crs.to_string().lower():
        gdf = gdf.to_crs("EPSG:4326")

    if climate_field not in gdf.columns:
        raise ValueError(f"Field '{climate_field}' not found in climate-zone shapefile.")

    cats = pd.Series(gdf[climate_field].astype(str).values)
    uniq = sorted(cats.unique().tolist())

    outlines = []
    names = []
    for c in uniq:
        geom = gdf.loc[gdf[climate_field].astype(str) == c].geometry.unary_union
        outlines.append(geom)
        names.append(c)

    regions = regionmask.Regions(outlines=outlines, names=names, numbers=list(range(len(names))))
    mask = regions.mask(ref["lon"], ref["lat"])  # (lat, lon): 0..K-1, NaN outside
    climate_id = mask.astype("float32") + 1.0    # 1..K, NaN outside

    mapping = {str(i + 1): names[i] for i in range(len(names))}
    return climate_id, mapping


def _choose_out_paths(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    out = cfg.get("output", {})
    model_dir = Path(out.get("model_dir", "./models")).expanduser().resolve()
    predict_dir = Path(out.get("predict_dir", "./predict")).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    predict_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "fusion_rf_twostage.joblib"
    fused_path = predict_dir / "pr_fused_rf_twostage.nc"
    return model_path, fused_path


def _open_inputs_v2(cfg: Dict[str, Any]):
    io_cfg = cfg.get("io", {})
    chunks = io_cfg.get("chunks", None)

    ref_info = cfg["reference"]
    ref_ds = open_ds(ref_info["path"], chunks=chunks)
    ref_var = ref_info.get("var", None) or list(ref_ds.data_vars.keys())[0]
    if ref_var not in ref_ds.data_vars:
        raise ValueError(f"Reference dataset missing variable '{ref_var}'")

    # precip products
    prods = []
    for p in cfg["precip_products"]:
        ds = open_ds(p["path"], chunks=chunks)
        var = p.get("var", None) or list(ds.data_vars.keys())[0]
        if var not in ds.data_vars:
            raise ValueError(f"Product {p.get('name','?')} missing var '{var}'")
        ds = align_grid(ds, ref_ds)
        prods.append((p.get("name", Path(p["path"]).stem), ds[var].astype("float32")))

    # static covariates (dict -> list)
    static_covs = []
    for name, item in (cfg.get("static_covariates", {}) or {}).items():
        ds = open_ds(item["path"], chunks=None)
        ds = align_grid(ds, ref_ds)
        var = item.get("var", None) or name or list(ds.data_vars.keys())[0]
        da = ds[var]
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)
        static_covs.append((name, da.load().astype("float32")))

    # dynamic covariates
    dynamic_covs = []
    for name, item in (cfg.get("dynamic_covariates", {}) or {}).items():
        ds = open_ds(item["path"], chunks=chunks)
        ds = align_grid(ds, ref_ds)
        var = item.get("var", None) or name or list(ds.data_vars.keys())[0]
        dynamic_covs.append((name, ds[var].astype("float32")))

    # climate mask
    cz = cfg["climate_zones"]
    climate_id, climate_mapping = build_climate_mask(cz["path"], cz["field"], ref_ds)

    # time intersection across ref + prods + dynamic
    ds_list = [ref_ds[[ref_var]]]
    for _, da in prods:
        ds_list.append(da.to_dataset(name="pr"))
    for _, da in dynamic_covs:
        ds_list.append(da.to_dataset(name="v"))

    ds_list_aligned, times = time_intersection(ds_list)

    ref_aligned = ds_list_aligned[0][ref_var].astype("float32")
    idx = 1

    prods_aligned = []
    for name, _ in prods:
        prods_aligned.append((name, ds_list_aligned[idx]["pr"].astype("float32")))
        idx += 1

    dyn_aligned = []
    for name, _ in dynamic_covs:
        dyn_aligned.append((name, ds_list_aligned[idx]["v"].astype("float32")))
        idx += 1

    return ref_ds, ref_aligned, prods_aligned, static_covs, dyn_aligned, climate_id, climate_mapping, times


def _select_models(cfg: Dict[str, Any]):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    tr = cfg.get("training", {})
    n_jobs = int(tr.get("n_jobs", -1))
    seed = int(tr.get("seed", 42))

    occ = (cfg.get("models", {}) or {}).get("occurrence", {}) or {}
    reg = (cfg.get("models", {}) or {}).get("regression", {}) or {}

    # occurrence classifier
    if bool(occ.get("use_catboost", False)):
        logger.warning("use_catboost=true but CatBoost is not supported in v1.0 fusion module; using RandomForestClassifier.")
    clf = RandomForestClassifier(
        n_estimators=int(occ.get("rf_n_estimators", 400)),
        n_jobs=n_jobs,
        random_state=seed,
        class_weight="balanced",
        min_samples_leaf=2,
    )

    # regression model
    reg_type = str(reg.get("type", "rf")).lower()
    if reg_type == "hgb":
        # Histogram GB (fast, no n_jobs in sklearn)
        reg_model = HistGradientBoostingRegressor(
            max_depth=int(reg.get("hgb_max_depth", 10)),
            learning_rate=float(reg.get("hgb_lr", 0.05)),
            max_iter=int(reg.get("hgb_max_iter", 500)),
            random_state=seed,
        )
    else:
        reg_model = RandomForestRegressor(
            n_estimators=int(reg.get("rf_n_estimators", 400)),
            n_jobs=n_jobs,
            random_state=seed,
            min_samples_leaf=2,
        )

    return clf, reg_model


@dataclass
class FusionRFTwoStageConfig:
    """Thin wrapper for the v2 JSON configuration."""
    cfg: Dict[str, Any]

    @staticmethod
    def from_json(path: PathLike) -> "FusionRFTwoStageConfig":
        p = Path(path).expanduser().resolve()
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return FusionRFTwoStageConfig(cfg=cfg)

    def to_json(self, path: PathLike) -> str:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=2)
        return str(p)


class RFTwoStageFuser:
    """Main entry: train and predict using per-climate two-stage models."""

    def __init__(self, config: Union[FusionRFTwoStageConfig, Dict[str, Any], str, PathLike]):
        _require_optional()

        if isinstance(config, FusionRFTwoStageConfig):
            self.cfg = config.cfg
        elif isinstance(config, (str, Path)):
            self.cfg = FusionRFTwoStageConfig.from_json(config).cfg
        elif isinstance(config, dict):
            self.cfg = config
        else:
            raise TypeError("config must be FusionRFTwoStageConfig | dict | path-to-json")

        lvl = (self.cfg.get("logging", {}) or {}).get("level", "INFO")
        logging.getLogger().setLevel(getattr(logging, str(lvl).upper(), logging.INFO))

        self.model_path, self.fused_path = _choose_out_paths(self.cfg)

    def train(self) -> str:
        """Train per-climate models and save a joblib bundle."""
        import joblib
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        ref_ds, ref_pr, prods, static_covs, dynamic_covs, climate_id, climate_mapping, times = _open_inputs_v2(self.cfg)

        tr = self.cfg.get("training", {})
        seed = int(tr.get("seed", 42))
        wet_thr = float(tr.get("wet_threshold_mmday", 0.1))
        wet_prob_thr = float(tr.get("wet_prob_threshold", 0.5))
        samples_per_climate = int(tr.get("samples_per_climate", 200000))
        min_samples_per_climate = int(tr.get("min_samples_per_climate", 50000))
        replace_cells = bool(tr.get("replace_cells", True))

        clf_base, reg_base = _select_models(self.cfg)

        clim_arr = climate_id.values
        valid = np.isfinite(clim_arr)
        if not valid.any():
            raise ValueError("No valid China cells from climate-zone mask.")
        clim_ids = np.unique(clim_arr[valid]).astype(int).tolist()

        models: Dict[str, Any] = {}
        diags: Dict[str, Any] = {}

        precip_names = [n for n, _ in prods]
        static_names = [n for n, _ in static_covs]
        dynamic_names = [n for n, _ in dynamic_covs]

        for cid in clim_ids:
            yy, xx = np.where(clim_arr == float(cid))
            if yy.size == 0:
                continue

            rng = np.random.default_rng(seed + cid)
            n = int(samples_per_climate)

            sel = rng.choice(yy.size, size=n, replace=replace_cells)
            y_idx = yy[sel]
            x_idx = xx[sel]
            c_idx = np.full(n, cid, dtype=np.int16)
            t_idx = rng.integers(0, len(times), size=n, endpoint=False)

            t = xr.DataArray(t_idx, dims="n")
            y = xr.DataArray(y_idx, dims="n")
            x = xr.DataArray(x_idx, dims="n")

            # precip products (assume >=2; typically 3)
            pvals = [da.isel(time=t, lat=y, lon=x).values.astype(np.float32) for _, da in prods]
            if len(pvals) < 2:
                raise ValueError("Need at least 2 precipitation products for fusion.")
            # support first three with derived stats; if >3, still keep mean/std on all
            p_stack = np.stack(pvals, axis=0)  # (P, n)
            pr_mean = np.nanmean(p_stack, axis=0)
            pr_std = np.nanstd(p_stack, axis=0)

            # pairwise diffs for first three (compat with original script)
            p1 = pvals[0]
            p2 = pvals[1]
            p3 = pvals[2] if len(pvals) >= 3 else pr_mean
            d12 = np.abs(p1 - p2)
            d13 = np.abs(p1 - p3)
            d23 = np.abs(p2 - p3)

            tt = pd.DatetimeIndex(times[t_idx])
            doy_sin, doy_cos, mon_sin, mon_cos = cyclical_time_features(tt)

            # static covs
            svals = [da2d.isel(lat=y, lon=x).values.astype(np.float32) for _, da2d in static_covs]
            # dynamic covs
            dvals = [da3d.isel(time=t, lat=y, lon=x).values.astype(np.float32) for _, da3d in dynamic_covs]

            latv = ref_pr["lat"].values[y_idx].astype(np.float32)
            lonv = ref_pr["lon"].values[x_idx].astype(np.float32)
            climv = c_idx.astype(np.float32)

            # Features: use first 3 products + stats + cyc + covariates + coords + climate_id
            feats = [p1, p2, p3, pr_mean, pr_std, d12, d13, d23, doy_sin, doy_cos, mon_sin, mon_cos]
            feats += svals
            feats += dvals
            feats += [latv, lonv, climv]

            X = np.stack(feats, axis=1).astype(np.float32)
            y_true = ref_pr.isel(time=t, lat=y, lon=x).values.astype(np.float32)

            m = np.isfinite(y_true)
            X = X[m]
            y_true = y_true[m]

            if X.shape[0] < min_samples_per_climate:
                logger.warning("Skip climate_id=%s: samples=%s < min_samples_per_climate=%s",
                               cid, X.shape[0], min_samples_per_climate)
                continue

            y_wet = (y_true >= wet_thr).astype(np.int32)

            X, med = median_impute(X, med=None)
            y_reg = np.log1p(np.maximum(y_true, 0.0)).astype(np.float32)

            # clone base models (simple re-init to avoid shared state)
            clf, reg = _select_models(self.cfg)
            clf.fit(X, y_wet)

            wet_idx = np.where(y_wet == 1)[0]
            if wet_idx.size < int(tr.get("min_wet_samples", 5000)):
                logger.warning("Skip climate_id=%s: wet samples=%s < min_wet_samples", cid, wet_idx.size)
                continue
            reg.fit(X[wet_idx], y_reg[wet_idx])

            # quick diagnostics (train-fit)
            proba = clf.predict_proba(X)[:, 1].astype(np.float32)
            wet_hat = (proba >= wet_prob_thr)
            y_hat = np.zeros_like(y_true, dtype=np.float32)
            if wet_hat.any():
                y_hat[wet_hat] = np.expm1(reg.predict(X[wet_hat])).astype(np.float32)
            y_hat = np.maximum(y_hat, 0.0)

            rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
            mae = float(mean_absolute_error(y_true, y_hat))
            r2 = float(r2_score(y_true, y_hat))

            key = str(cid)
            models[key] = {
                "impute_median": med,
                "clf": clf,
                "reg": reg,
                "diag": {"rmse": rmse, "mae": mae, "r2": r2},
                "climate_name": climate_mapping.get(key, f"climate_{cid}"),
                "feature_spec": {
                    "precip_products": precip_names,
                    "static_covariates": static_names,
                    "dynamic_covariates": dynamic_names,
                },
            }
            diags[key] = models[key]["diag"]

            logger.info("Trained climate_id=%s (%s): RMSE=%.3f MAE=%.3f R2=%.3f",
                        key, models[key]["climate_name"], rmse, mae, r2)

        bundle = {
            "wet_threshold_mmday": wet_thr,
            "wet_prob_threshold": wet_prob_thr,
            "climate_mapping": climate_mapping,
            "models": models,
            "train_diags": diags,
        }

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.model_path)

        # persist config used
        cfg_used_path = self.model_path.with_suffix(self.model_path.suffix + ".config_used.json")
        with open(cfg_used_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=2)

        logger.info("Saved fusion model: %s", str(self.model_path))
        return str(self.model_path)

    def predict(self) -> str:
        """Predict fused precipitation using trained per-climate models."""
        import joblib
        from tqdm import tqdm

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}. Run train() first.")

        bundle = joblib.load(self.model_path)
        models: Dict[str, Any] = bundle.get("models", {})
        if not models:
            raise ValueError("Loaded fusion model bundle has empty 'models'.")

        wet_prob_thr = float(bundle.get("wet_prob_threshold", 0.5))

        ref_ds, ref_pr, prods, static_covs, dynamic_covs, climate_id, climate_mapping, times = _open_inputs_v2(self.cfg)

        lat = ref_ds["lat"].values
        lon = ref_ds["lon"].values
        nT = len(times)
        nY = len(lat)
        nX = len(lon)

        compute = self.cfg.get("compute", {}) or {}
        predict_block_days = int(compute.get("predict_block_days", 30))
        predict_cell_block = int(compute.get("predict_cell_block", 60000))

        # derive block size from cell_block
        block_x = min(nX, max(40, int(np.sqrt(predict_cell_block))))
        block_y = min(nY, max(40, int(predict_cell_block // max(1, block_x))))

        io_cfg = self.cfg.get("io", {}) or {}
        clevel = int(io_cfg.get("netcdf_complevel", 4))
        mask_outside = True  # default per your script behavior

        clim2d = climate_id.values.astype(np.float32)
        static2d = {name: da.values.astype(np.float32) for name, da in static_covs}

        tmp_dir = self.fused_path.with_suffix(self.fused_path.suffix + ".tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        part_files: List[str] = []

        precip_names = [n for n, _ in prods]
        static_names = [n for n, _ in static_covs]
        dynamic_names = [n for n, _ in dynamic_covs]

        logger.info("Predict time range: %s to %s (n=%s)",
                    pd.to_datetime(times[0]).date(), pd.to_datetime(times[-1]).date(), nT)
        logger.info("Predict blocks: time=%s days; spatial=(%s,%s) cells; complevel=%s",
                    predict_block_days, block_y, block_x, clevel)

        for t_start in tqdm(range(0, nT, predict_block_days), desc="Predict (time batches)"):
            t_end = min(t_start + predict_block_days, nT)
            times_blk = pd.DatetimeIndex(times[t_start:t_end])
            T = len(times_blk)

            fused_blk = np.full((T, nY, nX), np.nan, dtype=np.float32)

            doy_sin, doy_cos, mon_sin, mon_cos = cyclical_time_features(times_blk)

            dyn_blk = {name: da.isel(time=slice(t_start, t_end)) for name, da in dynamic_covs}
            p_blk = {name: da.isel(time=slice(t_start, t_end)) for name, da in prods}

            for y0 in range(0, nY, block_y):
                y1 = min(y0 + block_y, nY)
                Yb = y1 - y0
                lat_blk_1d = lat[y0:y1].astype(np.float32)

                for x0 in range(0, nX, block_x):
                    x1 = min(x0 + block_x, nX)
                    Xb = x1 - x0
                    lon_blk_1d = lon[x0:x1].astype(np.float32)

                    clim_blk = clim2d[y0:y1, x0:x1]
                    valid_mask_2d = np.isfinite(clim_blk)
                    if not valid_mask_2d.any():
                        continue

                    # precip arrays (T,Yb,Xb) for first three products
                    p_arrays = []
                    for pname in precip_names[:3]:
                        p_arrays.append(p_blk[pname].isel(lat=slice(y0, y1), lon=slice(x0, x1)).load().values.astype(np.float32))
                    while len(p_arrays) < 3:
                        p_arrays.append(np.nanmean(np.stack(p_arrays, axis=0), axis=0))

                    p1, p2, p3 = p_arrays[0], p_arrays[1], p_arrays[2]
                    pr_mean = (p1 + p2 + p3) / 3.0
                    pr_std = np.nanstd(np.stack([p1, p2, p3], axis=0), axis=0)
                    d12 = np.abs(p1 - p2)
                    d13 = np.abs(p1 - p3)
                    d23 = np.abs(p2 - p3)

                    N = T * Yb * Xb

                    # time cyc (T,Yb,Xb)
                    def _expand_time(v):
                        vv = np.repeat(v[:, None, None], Yb, axis=1)
                        vv = np.repeat(vv, Xb, axis=2)
                        return vv.astype(np.float32)

                    doy_sin3 = _expand_time(doy_sin)
                    doy_cos3 = _expand_time(doy_cos)
                    mon_sin3 = _expand_time(mon_sin)
                    mon_cos3 = _expand_time(mon_cos)

                    # static -> (T,Yb,Xb)
                    s_list = []
                    for sn in static_names:
                        s2 = static2d[sn][y0:y1, x0:x1]
                        s3 = np.repeat(s2[None, :, :], T, axis=0)
                        s_list.append(s3.astype(np.float32))

                    # dynamic -> (T,Yb,Xb)
                    d_list = []
                    for dn in dynamic_names:
                        d3 = dyn_blk[dn].isel(lat=slice(y0, y1), lon=slice(x0, x1)).load().values.astype(np.float32)
                        d_list.append(d3)

                    # lat/lon -> (T,Yb,Xb)
                    lat2d = np.repeat(lat_blk_1d[:, None], Xb, axis=1).astype(np.float32)
                    lon2d = np.repeat(lon_blk_1d[None, :], Yb, axis=0).astype(np.float32)
                    lat3 = np.repeat(lat2d[None, :, :], T, axis=0)
                    lon3 = np.repeat(lon2d[None, :, :], T, axis=0)

                    clim3 = np.repeat(clim_blk[None, :, :], T, axis=0).astype(np.float32)

                    feats = [p1, p2, p3, pr_mean, pr_std, d12, d13, d23,
                             doy_sin3, doy_cos3, mon_sin3, mon_cos3]
                    feats += s_list
                    feats += d_list
                    feats += [lat3, lon3, clim3]

                    Xmat = np.stack(feats, axis=0).astype(np.float32)  # (F,T,Y,X)
                    Xmat = Xmat.reshape((Xmat.shape[0], N)).T          # (N,F)

                    clim_vec = Xmat[:, -1]
                    clim_int = np.where(np.isfinite(clim_vec), clim_vec.astype(np.int32), 0)

                    y_pred = np.full((N,), np.nan, dtype=np.float32)

                    uniq_cids = np.unique(clim_int)
                    for cid in uniq_cids:
                        if cid <= 0:
                            continue
                        key = str(cid)
                        if key not in models:
                            continue
                        m = (clim_int == cid)
                        if not m.any():
                            continue

                        med = np.array(models[key]["impute_median"], dtype=np.float32)
                        Xm, _ = median_impute(Xmat[m], med=med)

                        clf = models[key]["clf"]
                        reg = models[key]["reg"]

                        proba = clf.predict_proba(Xm)[:, 1].astype(np.float32)
                        wet_hat = (proba >= wet_prob_thr)

                        ym = np.zeros((Xm.shape[0],), dtype=np.float32)
                        if wet_hat.any():
                            ym[wet_hat] = np.expm1(reg.predict(Xm[wet_hat])).astype(np.float32)

                        y_pred[m] = np.maximum(ym, 0.0)

                    y_pred_3d = y_pred.reshape((T, Yb, Xb)).astype(np.float32)

                    if mask_outside:
                        for tt in range(T):
                            tmp = y_pred_3d[tt]
                            tmp[~valid_mask_2d] = np.nan
                            y_pred_3d[tt] = tmp

                    fused_blk[:, y0:y1, x0:x1] = y_pred_3d

            ds_out = xr.Dataset(
                data_vars={"pr_fused": (("time", "lat", "lon"), fused_blk)},
                coords={"time": times_blk, "lat": lat, "lon": lon},
            )
            ds_out["pr_fused"].attrs.update(units="mm/day", long_name="Fused daily precipitation (RF Two-stage, per-climate models)")

            part_path = tmp_dir / f"part_{t_start:06d}_{t_end:06d}.nc"
            ds_out.to_netcdf(
                part_path,
                encoding={"pr_fused": dict(zlib=True, complevel=clevel, dtype="float32")},
            )
            part_files.append(str(part_path))

        ds_all = xr.open_mfdataset(part_files, combine="by_coords")
        self.fused_path.parent.mkdir(parents=True, exist_ok=True)
        ds_all.to_netcdf(
            self.fused_path,
            encoding={"pr_fused": dict(zlib=True, complevel=clevel, dtype="float32")},
        )

        # cleanup
        for f in part_files:
            try:
                os.remove(f)
            except Exception:
                pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass

        logger.info("Saved fused NetCDF: %s", str(self.fused_path))
        return str(self.fused_path)


# Convenience functions
def train_rf_twostage(config: Union[FusionRFTwoStageConfig, Dict[str, Any], str, PathLike]) -> str:
    return RFTwoStageFuser(config).train()


def predict_rf_twostage(config: Union[FusionRFTwoStageConfig, Dict[str, Any], str, PathLike]) -> str:
    return RFTwoStageFuser(config).predict()
