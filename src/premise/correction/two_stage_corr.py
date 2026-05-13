# -*- coding: utf-8 -*-
"""
premise.correction.two_stage_rf

Two-stage Random Forest precipitation bias correction with monthly volume constraint.

Core idea (daily):
  1) RFClassifier -> wet/dry occurrence
  2) RFRegressor  -> residual (truth - product), trained on wet samples
  3) Apply correction only on predicted-wet pixels, clamp to >=0
  4) Monthly volume constraint: rescale corrected daily fields so monthly totals
     match reference totals (only when reference exists)

This module is adapted from user's Correction-V3.py implementation.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


def _require_sklearn() -> None:
    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # noqa: F401
    except Exception as e:
        raise ImportError(
            "TwoStageRFPrecipCorrector requires scikit-learn. "
            "Install optional deps, e.g., `pip install premise[correction]` "
            "or `pip install scikit-learn`."
        ) from e


def _open_ds(path: PathLike) -> xr.Dataset:
    return xr.open_dataset(path, engine="netcdf4", cache=False)


def _default_main_var(ds: xr.Dataset, *, prefer_time: bool) -> str:
    """Pick a main variable from Dataset (exclude typical CRS vars)."""
    ignore = {"spatial_ref", "crs", "grid_mapping", "Lambert_Conformal", "lambert_conformal_conic"}
    cands = []
    for v in ds.data_vars:
        if v in ignore:
            continue
        da = ds[v]
        if da.ndim < 2:
            continue
        if prefer_time and ("time" not in da.dims):
            continue
        cands.append((da.size, v))
    if not cands:
        cands = [(ds[v].size, v) for v in ds.data_vars if v not in ignore]
    return max(cands, key=lambda x: x[0])[1]


@dataclass
class TwoStageRFConfig:
    # Required
    ref_path: str
    dem_path: str

    # Optional covariates
    static_paths: Optional[Dict[str, str]] = None
    dynamic_paths: Optional[Dict[str, str]] = None

    # Sampling / thresholds
    sample_days_per_month: int = 60
    wet_threshold: float = 0.1
    sample_points_per_day: int = 20000
    max_wet_points_per_day: int = 8000

    # RF hyperparameters
    clf_n_estimators: int = 80
    clf_max_depth: int = 12
    reg_n_estimators: int = 80
    reg_max_depth: int = 12
    random_state: int = 42

    # Monthly scaling bounds
    monthly_scale_min: float = 0.3
    monthly_scale_max: float = 3.0

    # I/O / naming
    out_var_name: str = "prec_corrected_two_stage"
    ref_var: Optional[str] = None   # if None, auto-pick
    prod_var: Optional[str] = None  # if None, auto-pick

    @staticmethod
    def from_json(path: PathLike) -> "TwoStageRFConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return TwoStageRFConfig(**cfg)

    def to_json(self, path: PathLike) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return str(p)


class TwoStageRFPrecipCorrector:
    """
    Two-stage precipitation corrector with monthly volume constraint.

    Assumptions:
      - DataArray dims include (time, lat, lon) or compatible with xarray.interp_like.
      - Reference defines land mask (non-NaN at time=0).
    """

    def __init__(self, cfg: TwoStageRFConfig, *, verbose: bool = True):
        _require_sklearn()
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # local import

        self.cfg = cfg
        self.verbose = bool(verbose)

        self.rng = np.random.default_rng(int(cfg.random_state))
        self.sample_days_per_month = int(cfg.sample_days_per_month)
        self.wet_threshold = float(cfg.wet_threshold)
        self.sample_points_per_day = int(cfg.sample_points_per_day)
        self.max_wet_points_per_day = int(cfg.max_wet_points_per_day)
        self.monthly_scale_min = float(cfg.monthly_scale_min)
        self.monthly_scale_max = float(cfg.monthly_scale_max)

        self._RFClassifier = RandomForestClassifier
        self._RFRegressor = RandomForestRegressor

        self.clf_params = dict(
            n_estimators=int(cfg.clf_n_estimators),
            max_depth=int(cfg.clf_max_depth),
            n_jobs=-1,
            random_state=int(cfg.random_state),
            class_weight="balanced",
        )
        self.reg_params = dict(
            n_estimators=int(cfg.reg_n_estimators),
            max_depth=int(cfg.reg_max_depth),
            n_jobs=-1,
            random_state=int(cfg.random_state),
        )

        # ---- Reference (truth) ----
        self.ref_ds = _open_ds(cfg.ref_path)
        self.var_ref = cfg.ref_var or _default_main_var(self.ref_ds, prefer_time=True)

        ref_template = self.ref_ds[self.var_ref].isel(time=0).astype("float32")  # (lat,lon)
        self.lat = ref_template["lat"].values
        self.lon = ref_template["lon"].values
        self.nlat = self.lat.size
        self.nlon = self.lon.size

        lon2d, lat2d = np.meshgrid(self.lon, self.lat)
        self.lat_flat = lat2d.astype("float32").ravel()
        self.lon_flat = lon2d.astype("float32").ravel()

        land_mask_2d = ~np.isnan(ref_template.values)
        self.land_mask_flat = land_mask_2d.ravel()

        # ---- Static covariates ----
        self.static_data: Dict[str, np.ndarray] = {}

        dem_ds = _open_ds(cfg.dem_path)
        dem_var = _default_main_var(dem_ds, prefer_time=False)
        dem_da = dem_ds[dem_var].interp_like(ref_template, method="linear")
        self.static_data["DEM"] = self._as_2d_grid(dem_da, "DEM")

        if cfg.static_paths:
            for name, path in cfg.static_paths.items():
                ds = _open_ds(path)
                vname = _default_main_var(ds, prefer_time=False)
                da = ds[vname].interp_like(ref_template, method="linear")
                self.static_data[name] = self._as_2d_grid(da, name)

        # ---- Dynamic covariates ----
        self.dynamic_data: Dict[str, xr.DataArray] = {}
        if cfg.dynamic_paths:
            for name, path in cfg.dynamic_paths.items():
                ds = _open_ds(path)
                vname = _default_main_var(ds, prefer_time=True)
                da = ds[vname].interp_like(ref_template, method="linear").astype("float32")
                self.dynamic_data[name] = da

        if self.verbose:
            logger.info("TwoStageRF initialized.")
            logger.info("  ref_var=%s", self.var_ref)
            logger.info("  static=%s", list(self.static_data.keys()))
            logger.info("  dynamic=%s", list(self.dynamic_data.keys()))
            logger.info("  wet_threshold=%.3f", self.wet_threshold)

    def _as_2d_grid(self, da: xr.DataArray, name: str) -> np.ndarray:
        arr = np.asarray(da.values)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D grid, got shape={arr.shape}, dims={da.dims}")
        if arr.shape != (self.nlat, self.nlon):
            raise ValueError(
                f"{name} grid shape mismatch: {arr.shape} vs ref {(self.nlat, self.nlon)}. "
                "Please regrid to reference grid first."
            )
        return arr.astype("float32")

    def _prepare_features(
        self,
        prod_da: xr.DataArray,
        ref_da: Optional[xr.DataArray] = None,
        *,
        time_for_dyn=None,
        max_points: Optional[int] = None,
    ):
        p2d = prod_da.squeeze().values.astype("float32")
        p_flat = p2d.ravel()
        valid_mask = self.land_mask_flat & ~np.isnan(p_flat)

        if ref_da is not None:
            r2d = ref_da.squeeze().values.astype("float32")
            r_flat = r2d.ravel()
            valid_mask &= ~np.isnan(r_flat)
        else:
            r_flat = None

        if not np.any(valid_mask):
            return None, None, None, None

        idx_valid = np.where(valid_mask)[0]

        # training: sample pixels for memory control
        if (max_points is not None) and (idx_valid.size > max_points):
            if r_flat is not None:
                truth_valid = r_flat[idx_valid]
                wet_idx = idx_valid[truth_valid > self.wet_threshold]
                dry_idx = idx_valid[truth_valid <= self.wet_threshold]

                if wet_idx.size > self.max_wet_points_per_day:
                    wet_pick = self.rng.choice(wet_idx, size=self.max_wet_points_per_day, replace=False)
                else:
                    wet_pick = wet_idx

                n_total = int(max_points)
                n_dry = max(0, n_total - wet_pick.size)
                if dry_idx.size > n_dry and n_dry > 0:
                    dry_pick = self.rng.choice(dry_idx, size=n_dry, replace=False)
                else:
                    dry_pick = dry_idx

                idx_use = np.concatenate([wet_pick, dry_pick])
            else:
                idx_use = self.rng.choice(idx_valid, size=int(max_points), replace=False)
        else:
            idx_use = idx_valid

        feature_list = [p_flat[idx_use]]

        for _, arr2d in self.static_data.items():
            feature_list.append(arr2d.ravel()[idx_use])

        if self.dynamic_data and (time_for_dyn is not None):
            for name, da_dyn in self.dynamic_data.items():
                try:
                    da_t = da_dyn.sel(time=time_for_dyn, method="nearest")
                    feature_list.append(da_t.values.astype("float32").ravel()[idx_use])
                except Exception as e:
                    logger.warning("Dynamic covariate '%s' align failed at %s: %s", name, str(time_for_dyn), e)

        # coords
        feature_list.append(self.lat_flat[idx_use])
        feature_list.append(self.lon_flat[idx_use])

        X = np.stack(feature_list, axis=1).astype("float32")

        if r_flat is not None:
            truth_vals = r_flat[idx_use]
            sat_vals = p_flat[idx_use]
            wet_label = (truth_vals > self.wet_threshold).astype("int8")
            residual = (truth_vals - sat_vals).astype("float32")
        else:
            wet_label, residual = None, None

        return X, wet_label, residual, valid_mask

    def _apply_monthly_volume_constraint(self, da_month_corr: xr.DataArray, ref_month: xr.DataArray) -> xr.DataArray:
        if ref_month.time.size == 0:
            return da_month_corr

        eps = 1e-6
        ref_sum = ref_month.sum(dim="time")
        corr_sum = da_month_corr.sel(time=ref_month.time).sum(dim="time")
        scale = ref_sum / (corr_sum + eps)

        nearly_zero = (ref_sum < self.wet_threshold) & (corr_sum < self.wet_threshold)
        scale = scale.where(~nearly_zero, 1.0)
        scale = scale.clip(min=self.monthly_scale_min, max=self.monthly_scale_max)

        return da_month_corr * scale

    def correct_product(
        self,
        prod_path: PathLike,
        output_path: PathLike,
        *,
        start_year: int = 2000,
        end_year: int = 2020,
    ) -> str:
        """Correct one daily precipitation product NetCDF (time,lat,lon)."""
        prod_name = Path(prod_path).stem
        if self.verbose:
            logger.info("Correcting product: %s", prod_name)

        ds_prod = _open_ds(prod_path)
        var_prod = self.cfg.prod_var or _default_main_var(ds_prod, prefer_time=True)

        prod_aligned = ds_prod[var_prod].interp_like(self.ref_ds[self.var_ref], method="linear").astype("float32")
        prod_sel = prod_aligned.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        ref_sel = self.ref_ds[self.var_ref].sel(time=prod_sel.time).astype("float32")

        corrected_chunks = []

        for month in range(1, 13):
            month_mask = prod_sel.time.dt.month == month
            prod_month = prod_sel.sel(time=month_mask)
            if prod_month.time.size == 0:
                continue

            ref_month = ref_sel.sel(time=prod_month.time)

            all_times = prod_month.time.values
            n_sample_days = min(self.sample_days_per_month, len(all_times))
            if n_sample_days == 0:
                corrected_chunks.append(prod_month)
                continue

            # IMPORTANT: use self.rng for reproducibility
            sample_times = self.rng.choice(all_times, size=n_sample_days, replace=False)

            X_clf_list, y_clf_list = [], []
            X_reg_list, y_reg_list = [], []

            for t_val in sample_times:
                p_t = prod_month.sel(time=t_val)
                r_t = ref_month.sel(time=t_val)
                X_t, wet_label_t, residual_t, _ = self._prepare_features(
                    p_t, r_t, time_for_dyn=t_val, max_points=self.sample_points_per_day
                )
                if X_t is None:
                    continue

                X_clf_list.append(X_t)
                y_clf_list.append(wet_label_t)

                wet_idx = wet_label_t == 1
                if np.any(wet_idx):
                    X_reg_list.append(X_t[wet_idx])
                    y_reg_list.append(residual_t[wet_idx])

            if (not X_clf_list) or (not X_reg_list):
                # insufficient training -> keep original
                corrected_chunks.append(prod_month)
                continue

            X_clf = np.concatenate(X_clf_list, axis=0)
            y_clf = np.concatenate(y_clf_list, axis=0)
            X_reg = np.concatenate(X_reg_list, axis=0)
            y_reg = np.concatenate(y_reg_list, axis=0)

            if self.verbose:
                logger.info("Month %02d: clf=%s, reg=%s, features=%s",
                            month, f"{X_clf.shape[0]:,}", f"{X_reg.shape[0]:,}", X_reg.shape[1])

            clf = self._RFClassifier(**self.clf_params)
            clf.fit(X_clf, y_clf)

            reg = self._RFRegressor(**self.reg_params)
            reg.fit(X_reg, y_reg)

            # apply day-by-day
            month_corr_list = []
            for t_val in prod_month.time.values:
                p_t = prod_month.sel(time=t_val)
                X_pred, _, _, valid_mask = self._prepare_features(p_t, ref_da=None, time_for_dyn=t_val)

                p_vals = p_t.squeeze().values.astype("float32")
                if X_pred is None:
                    corr_vals = p_vals
                else:
                    prob_wet = clf.predict_proba(X_pred)[:, 1]
                    wet_pred = prob_wet > 0.5

                    residual_flat = np.zeros(p_vals.size, dtype="float32")
                    if np.any(wet_pred):
                        idx_valid = np.where(valid_mask)[0]
                        idx_wet = idx_valid[wet_pred]
                        y_pred = reg.predict(X_pred[wet_pred]).astype("float32")
                        residual_flat[idx_wet] = y_pred

                    corr_vals = (p_vals.ravel() + residual_flat).reshape(p_vals.shape)
                    corr_vals[corr_vals < 0] = 0.0

                da_corr = xr.DataArray(corr_vals, coords=p_t.squeeze().coords, dims=p_t.squeeze().dims)
                month_corr_list.append(da_corr)

            da_month_corr = xr.concat(month_corr_list, dim="time")
            da_month_final = self._apply_monthly_volume_constraint(da_month_corr, ref_month)
            corrected_chunks.append(da_month_final)

            del clf, reg, X_clf, y_clf, X_reg, y_reg, month_corr_list, da_month_corr, da_month_final

        if not corrected_chunks:
            raise RuntimeError("No corrected months produced. Please check inputs/time coverage.")

        da_all = xr.concat(corrected_chunks, dim="time").sortby("time")
        da_all.name = self.cfg.out_var_name
        da_all.attrs["comment"] = (
            f"Two-stage RF (wet/dry + residual) corrected product: {os.path.basename(str(prod_path))} "
            f"with monthly volume constraint"
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        da_all.to_netcdf(out, engine="netcdf4")
        if self.verbose:
            logger.info("Output: %s", str(out))
        return str(out)
