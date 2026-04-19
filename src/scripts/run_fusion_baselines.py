#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI wrapper for PREMISE baseline fusion.

Examples:
  python scripts/run_fusion_baselines.py fit      --config config_fusion_baselines.json
  python scripts/run_fusion_baselines.py fuse_all --config config_fusion_baselines.json
  python scripts/run_fusion_baselines.py fuse_one --config config_fusion_baselines.json --method inv_rmse_zone_month
"""

import argparse
import premise as pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "fuse_all", "fuse_one"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", default=None)
    args = ap.parse_args()

    if args.cmd == "fit":
        pm.fusion.fit_baseline_weights(args.config)
    elif args.cmd == "fuse_all":
        pm.fusion.fuse_baselines(args.config)
    else:
        if not args.method:
            raise ValueError("--method is required for fuse_one")
        pm.fusion.fuse_baselines(args.config, method=args.method)


if __name__ == "__main__":
    main()
