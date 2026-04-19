from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr


def open_dataset(path: str | Path, chunks: dict[str, int] | None = None) -> xr.Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    kwargs: dict[str, Any] = {}
    if chunks:
        kwargs["chunks"] = chunks
    return xr.open_dataset(path, **kwargs)


def save_dataset(ds: xr.Dataset, path: str | Path, comp_level: int = 4) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_file():
        path.unlink()

    encoding: dict[str, dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        if getattr(da.dtype, "kind", "") in "fiu":
            encoding[name] = {"zlib": True, "complevel": int(comp_level)}

    ds.load().to_netcdf(str(path), mode="w", encoding=encoding)
    return str(path)


def subset_time(ds: xr.Dataset, time_range: dict[str, str] | None = None) -> xr.Dataset:
    if time_range is None:
        return ds
    start = time_range.get("start")
    end = time_range.get("end")
    if "time" not in ds.coords and "time" not in ds.dims:
        raise KeyError("Dataset does not contain time dimension for time subsetting.")
    return ds.sel(time=slice(start, end))
