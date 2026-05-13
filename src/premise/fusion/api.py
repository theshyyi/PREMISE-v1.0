# -*- coding: utf-8 -*-
"""
premise.fusion.api
==================

High-level "one-call" runners that minimize user setup.

- run_baselines(config, methods=None, auto_fit=True)
- run_benchmarks(config, methods=None, auto_train=True, auto_geo=True)

These are designed for library users who prefer:
  pm.fusion.run_baselines("config.json")
  pm.fusion.run_benchmarks("config.json")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from ._common import as_config
from .baselines import fit_baseline_weights, fuse_baselines
from .benchmarks import fit_params, fit_ml, fit_geo_residual_month, predict


def run_baselines(
    config: Union[dict, str, Path],
    methods: Optional[Sequence[str]] = None,
    auto_fit: bool = True,
) -> Dict[str, str]:
    cfg = as_config(config)
    m = list(methods) if methods is not None else list(cfg["baseline"]["methods"])
    # auto fit if needed
    if auto_fit and any("zone_month" in x for x in m):
        fit_baseline_weights(cfg)
    out = {}
    for x in m:
        fuse_baselines(cfg, method=x)
        out[x] = str(Path(cfg["baseline"]["out_dir"]).expanduser().resolve() / x)
    return out


def run_benchmarks(
    config: Union[dict, str, Path],
    methods: Optional[Sequence[str]] = None,
    auto_train: bool = True,
    auto_geo: bool = True,
    base_method_for_geo: str = "evw_ref_zone_month",
) -> Dict[str, str]:
    cfg = as_config(config)
    m = list(methods) if methods is not None else list(cfg["benchmark"]["methods"])

    out_dir = Path(cfg["benchmark"]["write"]["out_dir"]).expanduser().resolve()
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if auto_train:
        fit_params(cfg)
        if any(x.startswith("ml_") for x in m):
            fit_ml(cfg)
    if auto_geo and ("geo_residual_idw_month" in m):
        fit_geo_residual_month(cfg, base_method=base_method_for_geo)

    # predict produces outputs for all methods in cfg["benchmark"]["methods"]
    predict(cfg)
    return {x: str(out_dir / x) for x in m}
