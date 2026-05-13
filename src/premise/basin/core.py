from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr


def _guess_engine(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == '.nc':
        for engine in ('h5netcdf', 'netcdf4', 'scipy'):
            try:
                if engine == 'h5netcdf':
                    import h5netcdf  # noqa: F401
                elif engine == 'netcdf4':
                    import netCDF4  # noqa: F401
                else:
                    import scipy  # noqa: F401
                return engine
            except Exception:
                continue
    return None


def open_dataset(path: str | Path, **kwargs: Any) -> xr.Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    engine = kwargs.pop('engine', None)
    if engine is None:
        engine = _guess_engine(path)

    if engine is None:
        return xr.open_dataset(path, **kwargs)
    return xr.open_dataset(path, engine=engine, **kwargs)


def save_dataset(ds: xr.Dataset, path: str | Path, *, encoding: dict | None = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file():
            path.unlink()
        else:
            raise IsADirectoryError(f"Output path exists but is not a file: {path}")

    engine = _guess_engine(path)
    kwargs: dict[str, Any] = {}
    if engine is not None:
        kwargs['engine'] = engine
    actual_encoding = encoding or {}
    if engine == 'scipy':
        kwargs['format'] = 'NETCDF3_64BIT'
        actual_encoding = {}

    ds.load().to_netcdf(str(path), mode='w', encoding=actual_encoding, **kwargs)
    return str(path)
