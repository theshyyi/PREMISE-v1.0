from __future__ import annotations

from typing import Any, Dict, Mapping

_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_product(name: str, spec: Mapping[str, Any], *, overwrite: bool = False) -> None:
    if name in _REGISTRY and not overwrite:
        raise KeyError(f"Product '{name}' already registered. Use overwrite=True to replace.")
    _REGISTRY[name] = dict(spec)


def get_product(name: str) -> Dict[str, Any]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown product '{name}'. Register it via premise.registry.register_product()."
        )
    return dict(_REGISTRY[name])


# ------------------------
# Built-in minimal examples
# ------------------------

# CHIRPS v3-like naming usually contains YYYY.MM.DD
register_product(
    "CHIRPSv3-sat",
    {
        "format": "geotiff",
        "glob_pattern": "*.tif",
        "regex_pattern": r"(?P<Y>\d{4})[._-](?P<M>\d{2})[._-](?P<D>\d{2})",
        "var_name": "pr",
        "units": "mm/day",
        "nodata": -9999.0,
    },
    overwrite=True,
)

# CPC_GLOBAL_DAILY: 需要你按实际产品补齐网格参数（下面是模板）
# 命中后用户可做到：pm.io.to_netcdf("xxx.bin", "xxx.nc", product="CPC_GLOBAL_DAILY")
# register_product(
#     "CPC_GLOBAL_DAILY",
#     {
#         "format": "raw",
#         "meta": {
#             "nx": 720,
#             "ny": 360,
#             "nt": 1,
#             "dtype": "float32",
#             "endian": "little",
#             "order": "tyx",
#             "lon_start": 0.25,
#             "lon_step": 0.5,
#             "lat_start": -89.75,
#             "lat_step": 0.5,
#             "missing_value": -9999,
#             "var_name": "pr",
#             "var_units": "mm/day",
#             "var_long_name": "Daily precipitation",
#         },
#     },
#     overwrite=True,
# )
