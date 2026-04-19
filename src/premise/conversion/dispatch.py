from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import xarray as xr

from .core.netcdf import open_nc, to_netcdf as save_nc
from .formats.binary import load_binary_to_dataarray
from .formats.geotiff import open_geotiff_stack
from .formats.grib import open_grib
from .formats.hdf import open_hdf
from .registry import get_product


def detect_format(path: str | Path) -> str:
    p = Path(path)
    if p.is_dir():
        if any(p.glob("*.tif")) or any(p.glob("*.tiff")):
            return "geotiff"
        if any(p.glob("*.nc")) or any(p.glob("*.nc4")):
            return "netcdf"
        if any(p.glob("*.grb*")) or any(p.glob("*.grib*")):
            return "grib"
        return "unknown_dir"

    suffixes = [s.lower() for s in p.suffixes]
    suffix = p.suffix.lower()
    if suffix in {".nc", ".nc4"}:
        return "netcdf"
    if suffix in {".tif", ".tiff"}:
        return "geotiff"
    if suffix in {".grb", ".grib", ".grb2", ".grib2"}:
        return "grib"
    if suffix in {".h5", ".hdf5", ".hdf", ".he5"}:
        return "hdf"
    if suffix in {".bin", ".dat", ".raw"} or suffixes[-2:] == [".nc", ".gz"]:
        return "raw"
    return "unknown"


def open_standard_dataset(
    path: str | Path,
    *,
    product: Optional[str] = None,
    var: Optional[str] = None,
    hints: Optional[Dict[str, Any]] = None,
    chunks: str | dict | None = "auto",
    decode_times: bool = True,
) -> xr.Dataset:
    p = Path(path)
    hints = hints or {}

    if product is not None:
        spec = get_product(product)
        fmt = spec.get("format")
    else:
        spec = {}
        fmt = detect_format(p)

    if fmt == "netcdf":
        ds = open_nc(p, chunks=chunks, decode_times=decode_times)
        return ds if (var is None) else ds[[var]]

    if fmt == "geotiff":
        ds = open_geotiff_stack(
            p,
            glob_pattern=spec.get("glob_pattern", hints.get("glob_pattern", "*.tif")),
            regex_pattern=spec.get("regex_pattern", hints.get("regex_pattern")),
            var_name=spec.get("var_name", hints.get("var_name", var or "pr")),
            units=spec.get("units", hints.get("units")),
            nodata=spec.get("nodata", hints.get("nodata")),
        )
        return ds if (var is None) else ds[[var]]

    if fmt == "raw":
        if "meta" in spec:
            meta = spec["meta"]
        elif "meta" in hints:
            meta = hints["meta"]
        elif "meta_path" in hints:
            meta = hints["meta_path"]
        else:
            raise ValueError("RAW binary needs spec['meta'] or hints['meta']/['meta_path']")
        da = load_binary_to_dataarray(meta, data_path=str(p))
        ds = da.to_dataset(name=da.name)
        return ds if (var is None) else ds[[var]]

    if fmt == "grib":
        return open_grib(
            p,
            var=var,
            filter_by_keys=hints.get("filter_by_keys"),
            backend_kwargs=hints.get("backend_kwargs"),
        )

    if fmt == "hdf":
        return open_hdf(
            p,
            group=hints.get("group"),
            dataset=hints.get("dataset"),
            var_name=hints.get("var_name", var or "var"),
            units=hints.get("units"),
            nodata=hints.get("nodata"),
            lat_path=hints.get("lat_path"),
            lon_path=hints.get("lon_path"),
            time_path=hints.get("time_path"),
        )

    raise ValueError(f"Unsupported input: {p} (format={fmt}).")


def save_standard_dataset(ds: xr.Dataset, path: str | Path, encoding: Optional[dict] = None) -> str:
    return save_nc(ds, path, encoding=encoding or {})


def convert_to_netcdf(
    path: str | Path,
    out_nc: str | Path,
    *,
    product: Optional[str] = None,
    var: Optional[str] = None,
    hints: Optional[Dict[str, Any]] = None,
    comp_level: int = 4,
) -> str:
    ds = open_standard_dataset(path, product=product, var=var, hints=hints)
    encoding: Dict[str, Dict[str, Any]] = {}
    for name in ds.data_vars:
        if ds[name].dtype.kind in "fiu":
            encoding[name] = {"zlib": True, "complevel": int(comp_level)}
    return save_nc(ds, out_nc, encoding=encoding)
