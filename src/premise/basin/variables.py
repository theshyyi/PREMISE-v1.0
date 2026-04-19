from __future__ import annotations

import xarray as xr


def rename_variables(data: xr.Dataset, mapping: dict[str, str] | None = None) -> xr.Dataset:
    if not mapping:
        return data
    missing = [k for k in mapping if k not in data.data_vars and k not in data.coords]
    if missing:
        raise KeyError(f"Variables/coords not found for rename: {missing}")
    return data.rename(mapping)
