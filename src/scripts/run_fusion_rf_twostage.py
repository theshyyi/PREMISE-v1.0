#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI wrapper for PREMISE RF two-stage precipitation fusion.

Train:
  python scripts/run_fusion_rf_twostage.py train --config config_fusion_rf.json

Predict:
  python scripts/run_fusion_rf_twostage.py predict --config config_fusion_rf.json
"""

import argparse
import premise as pm


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train")
    p1.add_argument("--config", required=True)

    p2 = sub.add_parser("predict")
    p2.add_argument("--config", required=True)

    args = ap.parse_args()

    if args.cmd == "train":
        pm.fusion.train_rf_twostage(args.config)
    else:
        pm.fusion.predict_rf_twostage(args.config)


if __name__ == "__main__":
    main()
