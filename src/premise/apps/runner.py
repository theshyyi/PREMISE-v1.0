from __future__ import annotations

from importlib import import_module
from typing import Any

from .common import load_config


APP_TYPE_TO_RUNNER = {
    'hydro_data_builder': ('premise.apps.hydro_data_builder', 'run_hydro_data_builder_task', 'run_hydro_data_builder_tasks'),
    'dataset_evaluator': ('premise.apps.dataset_evaluator', 'run_dataset_evaluator_task', 'run_dataset_evaluator_tasks'),
    'product_ranker': ('premise.apps.product_ranker', 'run_product_ranker_task', 'run_product_ranker_tasks'),
    'basin_forcing_case': ('premise.apps.basin_forcing_case', 'run_basin_forcing_case_task', 'run_basin_forcing_case_tasks'),
    'comparative_benchmark_case': ('premise.apps.comparative_benchmark_case', 'run_comparative_benchmark_case_task', 'run_comparative_benchmark_case_tasks'),
    'extremes_drought_case': ('premise.apps.extremes_drought_case', 'run_extremes_drought_case_task', 'run_extremes_drought_case_tasks'),
}


def _resolve_runners(app_type: str):
    if app_type not in APP_TYPE_TO_RUNNER:
        raise ValueError(f'Unsupported app_type: {app_type}')
    module_name, single_name, batch_name = APP_TYPE_TO_RUNNER[app_type]
    mod = import_module(module_name)
    return getattr(mod, single_name), getattr(mod, batch_name)


def run_application(config: dict[str, Any]):
    app_type = str(config['app_type'])
    single_runner, batch_runner = _resolve_runners(app_type)
    if 'tasks' in config:
        return batch_runner(config['tasks'])
    return single_runner(config)


def run_application_from_file(path: str):
    return run_application(load_config(path))
