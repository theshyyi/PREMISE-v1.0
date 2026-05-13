from __future__ import annotations

import xarray as xr


def clip_time(
    data: xr.Dataset | xr.DataArray,
    *,
    start: str | None = None,
    end: str | None = None,
    time_name: str = 'time',
) -> xr.Dataset | xr.DataArray:
    if time_name not in data.coords and time_name not in data.dims:
        raise KeyError(f"Time coordinate '{time_name}' not found")

    if start is None and end is None:
        return data
    if start is None:
        return data.sel({time_name: slice(None, end)})
    if end is None:
        return data.sel({time_name: slice(start, None)})
    return data.sel({time_name: slice(start, end)})
