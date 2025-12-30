#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run grid-based multi-product detection metrics against a reference dataset.

Example usage:

    python scripts/run_grid_evaluation.py \
        --obs /home/.../CMFDV2.TIMEFIX.daily.CHINA.nc \
        --sim-dir /home/.../PRE_MERGE/TIMEFIX/ \
        --out /home/.../detection_metrics.csv \
        --var pr \
        --threshold 1.0 \
        --pattern "*.TIMEFIX.daily.CHINA.nc" \
        --ref-prefix CMFDV2
"""

import argparse

from premise.workflows import compute_detection_metrics_for_products


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid-based detection metrics for multi-source precipitation products."
    )
    parser.add_argument("--obs", required=True, help="Reference NetCDF file.")
    parser.add_argument("--sim-dir", required=True, help="Directory of product NetCDF files.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--var", default="pr", help="Variable name in NetCDF (default: pr).")
    parser.add_argument(
        "--threshold", type=float, default=1.0, help="Precipitation threshold (mm/day)."
    )
    parser.add_argument(
        "--pattern",
        default="*.TIMEFIX.daily.CHINA.nc",
        help="Glob pattern for product files.",
    )
    parser.add_argument(
        "--ref-prefix",
        default=None,
        help="Prefix of reference product to be skipped in product list.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    compute_detection_metrics_for_products(
        obs_path=args.obs,
        sim_dir=args.sim_dir,
        var_name=args.var,
        threshold=args.threshold,
        pattern=args.pattern,
        ref_product_prefix=args.ref_prefix,
        out_csv=args.out,
    )


if __name__ == "__main__":
    main()
