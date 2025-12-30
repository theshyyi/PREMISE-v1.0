#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI wrapper for PREMISE benchmark/ablation fusion suite.

Examples:
  python scripts/run_fusion_benchmarks.py fit_params --config config_fusion_bench.json
  python scripts/run_fusion_benchmarks.py fit_ml     --config config_fusion_bench.json
  python scripts/run_fusion_benchmarks.py fit_geo    --config config_fusion_bench.json --base_method evw_ref_zone_month
  python scripts/run_fusion_benchmarks.py predict    --config config_fusion_bench.json
"""

import argparse
import premise as pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit_params", "fit_ml", "fit_geo", "predict"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_method", default="evw_ref_zone_month")
    args = ap.parse_args()

    if args.cmd == "fit_params":
        pm.fusion.fit_params(args.config)
    elif args.cmd == "fit_ml":
        pm.fusion.fit_ml(args.config)
    elif args.cmd == "fit_geo":
        pm.fusion.fit_geo_residual_month(args.config, base_method=args.base_method)
    else:
        pm.fusion.predict(args.config)


if __name__ == "__main__":
    main()
