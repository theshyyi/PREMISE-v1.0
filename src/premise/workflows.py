# -*- coding: utf-8 -*-

"""
workflows
=========

High-level workflows that orchestrate core modules:

- Grid-based multi-product detection metrics against a reference
- Index computation + regional aggregation (helper)
"""

from __future__ import annotations

import glob
import os
from typing import Optional, Sequence

import pandas as pd
import xarray as xr

from .metrics import pod, far, csi, fbias
from .indices import calc_spi, calc_spei, calc_sri, calc_sti
from .preprocess import area_mean_by_region

from premise.indices import calc_spi, calc_spei, calc_sri, calc_sti

def compute_detection_metrics_for_products(
    obs_path: str,
    sim_dir: str,
    var_name: str = "pr",
    threshold: float = 1.0,
    pattern: str = "*.TIMEFIX.daily.CHINA.nc",
    ref_product_prefix: Optional[str] = None,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Batch compute POD, FAR, CSI, and FBIAS for multiple products
    against a reference dataset.
    """
    ds_obs = xr.open_dataset(obs_path)
    if var_name not in ds_obs:
        raise KeyError(f"Variable '{var_name}' not found in obs file: {obs_path}")
    obs = ds_obs[var_name]

    nc_paths = sorted(glob.glob(os.path.join(sim_dir, pattern)))
    if not nc_paths:
        raise FileNotFoundError(f"No files matched pattern {pattern} in directory {sim_dir}")

    rows = []

    for nc_path in nc_paths:
        fname = os.path.basename(nc_path)

        if ref_product_prefix and fname.startswith(ref_product_prefix):
            continue

        product_name = fname.split(".TIMEFIX")[0]
        print(f"[premise] Detection metrics for product: {product_name}")

        ds_sim = xr.open_dataset(nc_path)
        if var_name not in ds_sim:
            print(f"  - WARNING: {var_name} not in {fname}, skip.")
            ds_sim.close()
            continue

        sim = ds_sim[var_name]

        sim, obs_aligned = xr.align(sim, obs, join="inner")

        pod_val = pod(obs_aligned, sim, threshold)
        far_val = far(obs_aligned, sim, threshold)
        csi_val = csi(obs_aligned, sim, threshold)
        fbias_val = fbias(obs_aligned, sim, threshold)

        rows.append(
            {
                "product": product_name,
                "threshold": threshold,
                "POD": pod_val,
                "FAR": far_val,
                "CSI": csi_val,
                "FBIAS": fbias_val,
            }
        )

        ds_sim.close()

    ds_obs.close()

    if not rows:
        raise RuntimeError("No valid products processed; please check directory and pattern.")

    df = pd.DataFrame(rows).sort_values("product").reset_index(drop=True)

    if out_csv is not None:
        out_dir = os.path.dirname(out_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[premise] Detection metrics saved to: {out_csv}")

    return df


def compute_indices_and_regional_means(
    ds: xr.Dataset,
    *,
    precip_var: str = "pr",
    temp_var: Optional[str] = None,
    pet_var: Optional[str] = None,
    runoff_var: Optional[str] = None,
    shp_path: str,
    region_field: Optional[str] = None,
    spi_scales: Sequence[int] = (3, 6, 12),
    spei_scales: Sequence[int] = (3, 6, 12),
    sri_scales: Sequence[int] = (),
    sti_scales: Sequence[int] = (),
) -> xr.Dataset:
    """
    Helper workflow: compute SPI/SPEI/SRI/STI and regional means.

    Returns a Dataset with variables:
        - SPI_{scale}
        - SPEI_{scale}
        - SRI_{scale}
        - STI_{scale}
        and additional variables:
        - SPI_{scale}_region
        - SPEI_{scale}_region
        etc. (time x region).
    """
    out_ds = xr.Dataset()

    pr = ds[precip_var]

    # SPI
    for sc in spi_scales:
        spi = calc_spi(pr, scale=sc)
        out_ds[f"SPI_{sc}"] = spi
        out_ds[f"SPI_{sc}_region"] = area_mean_by_region(
            spi, shp_path, region_field=region_field
        )

    # SPEI
    if pet_var is not None:
        pet = ds[pet_var]
        for sc in spei_scales:
            spei = calc_spei(pr, pet, scale=sc)
            out_ds[f"SPEI_{sc}"] = spei
            out_ds[f"SPEI_{sc}_region"] = area_mean_by_region(
                spei, shp_path, region_field=region_field
            )

    # SRI
    if runoff_var is not None:
        q = ds[runoff_var]
        for sc in sri_scales:
            sri = calc_sri(q, scale=sc)
            out_ds[f"SRI_{sc}"] = sri
            out_ds[f"SRI_{sc}_region"] = area_mean_by_region(
                sri, shp_path, region_field=region_field
            )

    # STI
    if temp_var is not None:
        tas = ds[temp_var]
        for sc in sti_scales:
            sti = calc_sti(tas, scale=sc)
            out_ds[f"STI_{sc}"] = sti
            out_ds[f"STI_{sc}_region"] = area_mean_by_region(
                sti, shp_path, region_field=region_field
            )

    return out_ds
