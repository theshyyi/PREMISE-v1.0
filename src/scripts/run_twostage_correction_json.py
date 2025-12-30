#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import premise as pm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    products = cfg.pop("products")
    out_dir = Path(cfg.pop("out_dir"))
    start_year = int(cfg.pop("start_year", 2000))
    end_year = int(cfg.pop("end_year", 2020))

    corr_cfg = pm.correction.TwoStageRFConfig(**cfg)
    corr = pm.correction.TwoStageRFPrecipCorrector(corr_cfg, verbose=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for p in products:
        p = Path(p)
        out_nc = out_dir / f"{p.stem}_Corrected_TwoStage.nc"
        corr.correct_product(p, out_nc, start_year=start_year, end_year=end_year)

if __name__ == "__main__":
    main()
