# -*- coding: utf-8 -*-
"""premise.fusion: precipitation fusion modules (PREMISE v1.0)."""

from .rf_twostage import (
    FusionRFTwoStageConfig,
    RFTwoStageFuser,
    train_rf_twostage,
    predict_rf_twostage,
)

from .baselines import (
    FusionBaselinesConfig,
    fit_baseline_weights,
    fuse_baselines,
)

from .benchmarks import (
    FusionBenchmarksConfig,
    fit_params,
    fit_ml,
    fit_geo_residual_month,
    predict,
)

from .api import (
    run_baselines,
    run_benchmarks,
)

__all__ = [
    # RF two-stage
    "FusionRFTwoStageConfig",
    "RFTwoStageFuser",
    "train_rf_twostage",
    "predict_rf_twostage",
    # Baselines
    "FusionBaselinesConfig",
    "fit_baseline_weights",
    "fuse_baselines",
    # Benchmarks
    "FusionBenchmarksConfig",
    "fit_params",
    "fit_ml",
    "fit_geo_residual_month",
    "predict",
    # One-call runners
    "run_baselines",
    "run_benchmarks",
]
