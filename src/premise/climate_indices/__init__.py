"""Climate index module for extreme and hydroclimatic indices."""

from .api import (
    compute_extreme_indices,
    compute_hydroclimatic_indices,
    run_climate_indices_task,
    run_climate_indices_tasks,
)
from .drought import calc_spi, calc_spei, calc_sri, calc_sti
from .extremes import rx1day, rx5day, prcptot, sdii, r10mm, r20mm, r95p, r99p, cdd, cwd

__all__ = [
    "compute_extreme_indices",
    "compute_hydroclimatic_indices",
    "run_climate_indices_task",
    "run_climate_indices_tasks",
    "calc_spi",
    "calc_spei",
    "calc_sri",
    "calc_sti",
    "rx1day",
    "rx5day",
    "prcptot",
    "sdii",
    "r10mm",
    "r20mm",
    "r95p",
    "r99p",
    "cdd",
    "cwd",
]
