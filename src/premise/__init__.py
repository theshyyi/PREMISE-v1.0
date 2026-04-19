"""PREMISE public package interface.

The top-level package keeps imports light so that optional geospatial,
plotting, or machine-learning dependencies are not required for a basic
``import premise``. The package is organized around application-oriented
modules, with data-format conversion fully integrated under
``premise.conversion``.
"""

from importlib import import_module

__version__ = "1.1.0"

_LAZY_SUBMODULES = {
    "acquisition",
    "conversion",
    "basin",
    "product_evaluation",
    "climate_indices",
    "visualization",
    "preprocess",
    "indices",
    "extreme_indices",
    "metrics",
    "evaluation",
    "plotting",
    "pointpixel",
    "workflows",
    "correction",
    "fusion",
}

__all__ = sorted(_LAZY_SUBMODULES | {"__version__"})


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        return import_module(f"premise.{name}")
    raise AttributeError(f"module 'premise' has no attribute {name!r}")
