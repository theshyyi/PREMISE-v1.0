from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import xarray as xr

from . import registry


def _sniff_format(p: Path) -> str:
    # directory sniff
    if p.is_dir():
        if any(p.glob("*.tif")) or any(p.glob("*.tiff")):
            return "geotiff"
        if any(p.glob("*.nc")) or any(p.glob("*.nc4")):
            return "netcdf"
        if any(p.glob("*.grb*")) or any(p.glob("*.grib*")):
            return "grib"
        return "unknown_dir"

    # file sniff by suffix (v1.0 minimal)
    suf = p.suffix.lower()
    if suf in [".nc", ".nc4"]:
        return "netcdf"
    if suf in [".tif", ".tiff"]:
        return "geotiff"
    if suf in [".grb", ".grib", ".grb2", ".grib2"]:
        return "grib"
    if suf in [".bin", ".dat", ".raw", ".gz"]:
        return "raw"
    if suf in [".h5", ".hdf5", ".hdf", ".he5", ".nc4"]:
        return "hdf"

    return "unknown"


def open_dataset(
    path: str | Path,
    *,
    product: Optional[str] = None,
    var: Optional[str] = None,
    hints: Optional[Dict[str, Any]] = None,
) -> xr.Dataset:
    """
    Unified open function:
      - NetCDF: xr.open_dataset
      - GeoTIFF: open_geotiff_stack
      - GRIB: xr.open_dataset(engine="cfgrib") (optional dependency)
      - RAW: binaryio.load_binary_to_dataarray (requires product meta or hints)
    """
    p = Path(path)
    hints = hints or {}

    if product is not None:
        spec = registry.get_product(product)
        fmt = spec.get("format")
    else:
        spec = {}
        fmt = _sniff_format(p)

    if fmt == "netcdf":
        ds = xr.open_dataset(p)
        return ds if (var is None) else ds[[var]]

    if fmt == "geotiff":
        from .readers.geotiff import open_geotiff_stack

        ds = open_geotiff_stack(
            p,
            glob_pattern=spec.get("glob_pattern", hints.get("glob_pattern", "*.tif")),
            regex_pattern=spec.get("regex_pattern", hints.get("regex_pattern", None)),
            var_name=spec.get("var_name", hints.get("var_name", "pr")),
            units=spec.get("units", hints.get("units", "mm/day")),
            nodata=float(spec.get("nodata", hints.get("nodata", -9999.0))),
        )
        return ds if (var is None) else ds[[var]]

    if fmt == "raw":
        from .readers.binaryio import load_binary_to_dataarray

        # 1) registry provides meta dict
        if "meta" in spec:
            da = load_binary_to_dataarray(spec["meta"], data_path=str(p))
            ds = da.to_dataset(name=da.name)
            return ds if (var is None) else ds[[var]]

        # 2) user provided meta_path or meta dict in hints
        if "meta" in hints:
            da = load_binary_to_dataarray(hints["meta"], data_path=str(p))
            ds = da.to_dataset(name=da.name)
            return ds if (var is None) else ds[[var]]

        if "meta_path" in hints:
            da = load_binary_to_dataarray(hints["meta_path"], data_path=str(p))
            ds = da.to_dataset(name=da.name)
            return ds if (var is None) else ds[[var]]

        raise ValueError(
            "RAW binary needs either registry meta (product spec['meta']) "
            "or hints['meta'] / hints['meta_path']."
        )

    if fmt == "grib":
        try:
            ds = xr.open_dataset(p, engine="cfgrib")
        except Exception as e:
            raise ImportError(
                "GRIB support requires cfgrib+eccodes. "
                "Install optional dependencies, e.g. `pip install premise[grib]`."
            ) from e
        return ds if (var is None) else ds[[var]]

    if fmt == "grib":
        from .readers.grib import open_grib
        return open_grib(p, var=var, filter_by_keys=hints.get("filter_by_keys"),
                         backend_kwargs=hints.get("backend_kwargs"))

    if fmt == "hdf":
        from .readers.hdf import open_hdf
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

    raise ValueError(f"Unsupported input: {p} (format={fmt}). Provide product or hints.")


def to_netcdf(
    path: str | Path,
    out_nc: str | Path,
    *,
    product: Optional[str] = None,
    var: Optional[str] = None,
    hints: Optional[Dict[str, Any]] = None,
    comp_level: int = 4,
) -> str:
    ds = open_dataset(path, product=product, var=var, hints=hints)

    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    # generic compression for numeric vars
    encoding: Dict[str, Dict[str, Any]] = {}
    for v in ds.data_vars:
        if ds[v].dtype.kind in "fiu":
            encoding[v] = {"zlib": True, "complevel": int(comp_level)}

    ds.to_netcdf(out_nc, format="NETCDF4", engine="netcdf4", encoding=encoding)
    return str(out_nc)
