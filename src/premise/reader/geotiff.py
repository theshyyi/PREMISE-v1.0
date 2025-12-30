# -*- coding: utf-8 -*-
"""
geotiff
=======

GeoTIFF -> NetCDF tools for daily precipitation stacks.

Key features for PREMISE v1.0:
- Lazy import rioxarray to avoid hard dependency at import time
- open_geotiff_stack(): open a GeoTIFF directory (daily files) into an xr.Dataset
- convert_daily_geotiff_to_monthly_nc(): keep your original batch conversion workflow
"""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import xarray as xr


def _require_rioxarray():
    try:
        import rioxarray as rxr  # noqa
        return rxr
    except Exception as e:
        raise ImportError(
            "GeoTIFF support requires rioxarray (and rasterio). "
            "Install optional dependencies, e.g. `pip install premise[tiff]`."
        ) from e


def _compile_ymd_pattern(pattern: str) -> re.Pattern:
    return re.compile(pattern)


def _auto_pick_ymd_pattern(file_names: Sequence[str]) -> re.Pattern:
    """
    Try multiple common patterns and pick the one matching the most filenames.
    """
    candidates = [
        r"(?P<Y>\d{4})[._-](?P<M>\d{2})[._-](?P<D>\d{2})",  # 2000.01.01 or 2000-01-01 or 2000_01_01
        r"(?P<Y>\d{4})(?P<M>\d{2})(?P<D>\d{2})",            # 20000101
    ]
    best_pat = None
    best_score = -1
    for pat in candidates:
        cre = re.compile(pat)
        score = sum(1 for fn in file_names if cre.search(fn) is not None)
        if score > best_score:
            best_score = score
            best_pat = cre
    if best_pat is None or best_score <= 0:
        raise ValueError(
            "Failed to auto-detect date pattern from filenames. "
            "Please provide regex_pattern explicitly."
        )
    return best_pat


def parse_ymd_from_name(fname: str, pattern: re.Pattern) -> Tuple[int, int, int]:
    """
    Parse (year, month, day) from filename using a compiled regex pattern.
    The regex must provide named groups: Y, M, D.
    """
    m = pattern.search(fname)
    if not m:
        raise ValueError(f"Cannot parse date from filename: {fname}")
    y = int(m.group("Y"))
    mo = int(m.group("M"))
    d = int(m.group("D"))
    return y, mo, d


def open_geotiff_stack(
    path: str | Path,
    *,
    glob_pattern: str = "*.tif",
    regex_pattern: Optional[str] = None,
    var_name: str = "pr",
    units: str = "mm/day",
    nodata: float = -9999.0,
) -> xr.Dataset:
    """
    Open a GeoTIFF file or a directory of daily GeoTIFFs into an xr.Dataset with dims (time, lat, lon).

    Parameters
    ----------
    path : file or directory
    glob_pattern : used when path is a directory
    regex_pattern : optional; if None, auto-detect common YYYY.MM.DD / YYYYMMDD patterns
    """
    rxr = _require_rioxarray()
    p = Path(path)

    if p.is_file():
        files = [str(p)]
        names_for_detect = [p.name]
    else:
        # directory
        files = sorted([str(x) for x in p.glob(glob_pattern)])
        # also try *.tiff if not found
        if not files and glob_pattern == "*.tif":
            files = sorted([str(x) for x in p.glob("*.tiff")])
        if not files:
            raise FileNotFoundError(f"No GeoTIFF files found in {p} with pattern {glob_pattern}")
        names_for_detect = [Path(x).name for x in files]

    pat = _compile_ymd_pattern(regex_pattern) if regex_pattern else _auto_pick_ymd_pattern(names_for_detect)

    das: List[xr.DataArray] = []
    times: List[np.datetime64] = []

    for fp in files:
        fname = Path(fp).name
        y, mo, d = parse_ymd_from_name(fname, pat)
        times.append(np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}"))

        da = rxr.open_rasterio(fp, masked=True).squeeze("band", drop=True)
        # explicit nodata handling (in case masked is not enough)
        da = da.where(da != nodata)

        # rename dims
        if "x" in da.dims:
            da = da.rename({"x": "lon"})
        if "y" in da.dims:
            da = da.rename({"y": "lat"})

        das.append(da)

    da_all = xr.concat(das, dim="time")
    da_all = da_all.assign_coords(time=("time", np.array(times, dtype="datetime64[ns]")))

    # ensure lat ascending
    if np.any(np.diff(da_all["lat"].values) < 0):
        da_all = da_all.sortby("lat")

    da_all = da_all.astype("float32")
    da_all.name = var_name
    da_all.attrs.update({"long_name": var_name, "units": units})

    da_all["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    da_all["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})

    return da_all.to_dataset()


def list_year_files(in_root: Path, year: int, *, glob_pattern: str) -> List[str]:
    """
    List files for a specific year directory or year-matching file set.
    """
    year_dir = in_root / f"{year}"
    if year_dir.exists() and year_dir.is_dir():
        files = sorted([str(p) for p in year_dir.glob(glob_pattern)])
    else:
        # fallback: search under in_root (some products store all years together)
        files = sorted(glob.glob(str(in_root / f"*{year}*")))
        files = [f for f in files if Path(f).suffix.lower() in [".tif", ".tiff"]]
    return files


def build_month_from_geotiffs(
    year: int,
    month: int,
    files: List[str],
    out_dir: Path,
    *,
    name_pattern: re.Pattern,
    var_name: str,
    units: str,
    nodata: float,
    comp_level: int,
    product_long_name: str = "",
    product_source: str = "",
) -> None:
    """
    Stack daily GeoTIFFs for a given (year, month) into a NetCDF (time, lat, lon).
    """
    if not files:
        return

    rxr = _require_rioxarray()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / f"{var_name}_daily_{year}_{month:02d}.nc"

    das: List[xr.DataArray] = []
    times: List[np.datetime64] = []

    for fp in sorted(files):
        fname = Path(fp).name
        y, mo, d = parse_ymd_from_name(fname, name_pattern)
        times.append(np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}"))

        da = rxr.open_rasterio(fp, masked=True).squeeze("band", drop=True)
        da = da.where(da != nodata)

        if "x" in da.dims:
            da = da.rename({"x": "lon"})
        if "y" in da.dims:
            da = da.rename({"y": "lat"})

        das.append(da)

    da_all = xr.concat(das, dim="time")
    da_all = da_all.assign_coords(time=("time", np.array(times, dtype="datetime64[ns]")))

    if np.any(np.diff(da_all["lat"].values) < 0):
        da_all = da_all.sortby("lat")

    da_all = da_all.astype("float32")
    da_all.name = var_name
    da_all.attrs.update(
        {
            "long_name": product_long_name or var_name,
            "units": units,
            "source": product_source,
        }
    )
    da_all["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    da_all["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})

    ds = da_all.to_dataset()

    encoding: Dict[str, Dict] = {
        var_name: {
            "zlib": True,
            "complevel": int(comp_level),
            "dtype": "float32",
            "_FillValue": np.float32(nodata),
        }
    }

    ds.to_netcdf(out_nc, format="NETCDF4", engine="netcdf4", encoding=encoding)


def convert_daily_geotiff_to_monthly_nc(
    in_root: str | Path,
    out_root: str | Path,
    year_beg: int,
    year_end: int,
    *,
    regex_pattern: str,
    glob_pattern: str,
    var_name: str = "pr",
    units: str = "mm/day",
    nodata: float = -9999.0,
    comp_level: int = 4,
    product_long_name: str = "",
    product_source: str = "",
) -> None:
    """
    Batch convert daily GeoTIFFs into monthly NetCDF files.

    You provide:
      - in_root: root dir (containing year subdirs or all files)
      - out_root: output dir
      - year_beg/year_end
      - regex_pattern: must define named groups Y/M/D
      - glob_pattern: e.g., "chirps-v3.0.sat.*.tif"
    """
    in_root = Path(in_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pat = _compile_ymd_pattern(regex_pattern)

    for year in range(year_beg, year_end + 1):
        files = list_year_files(in_root, year, glob_pattern=glob_pattern)
        if not files:
            print(f"[WARN] no files for year {year}")
            continue

        by_month: Dict[int, List[str]] = {m: [] for m in range(1, 13)}
        for fp in files:
            fname = Path(fp).name
            y, mo, _ = parse_ymd_from_name(fname, pat)
            if y != year:
                continue
            by_month[mo].append(fp)

        for month in range(1, 13):
            month_files = by_month[month]
            if not month_files:
                continue
            build_month_from_geotiffs(
                year=year,
                month=month,
                files=month_files,
                out_dir=out_root / f"{year}",
                name_pattern=pat,
                var_name=var_name,
                units=units,
                nodata=nodata,
                comp_level=comp_level,
                product_long_name=product_long_name,
                product_source=product_source,
            )
