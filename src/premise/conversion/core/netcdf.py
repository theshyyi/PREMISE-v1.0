from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import xarray as xr


def _choose_netcdf_engine() -> str | None:
    """
    Windows 下优先避免 netcdf4 写中文路径时的潜在兼容性问题，
    先尝试 h5netcdf，再尝试 scipy，最后才回退到 netcdf4。
    """
    for engine in ("h5netcdf", "scipy", "netcdf4"):
        try:
            if engine == "netcdf4":
                import netCDF4  # noqa: F401
            elif engine == "h5netcdf":
                import h5netcdf  # noqa: F401
            elif engine == "scipy":
                import scipy  # noqa: F401
            return engine
        except Exception:
            continue
    return None


def open_nc(
    path: str | Path,
    *,
    chunks: str | dict | None = "auto",
    decode_times: bool = True,
) -> xr.Dataset:
    engine = _choose_netcdf_engine()
    kwargs = {"engine": engine} if engine is not None else {}
    return xr.open_dataset(str(Path(path)), chunks=chunks, decode_times=decode_times, **kwargs)


def to_netcdf(
    ds: xr.Dataset,
    path: str | Path,
    encoding: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # 如果已有同名文件，先删除，避免 Windows 下覆盖写入时被锁导致 PermissionError
    if path.exists():
        if path.is_file():
            try:
                path.unlink()
            except PermissionError as e:
                raise PermissionError(
                    f"无法覆盖输出文件：{path}\n"
                    f"它可能正被其他程序占用（如 Python、Panoply、ArcGIS、VS Code 等）。"
                ) from e
        else:
            raise IsADirectoryError(f"输出路径已存在但不是文件：{path}")

    kwargs: Dict[str, Any] = {}
    engine = _choose_netcdf_engine()
    if engine is not None:
        kwargs["engine"] = engine

    # scipy 不支持 netCDF4 风格的压缩 encoding，直接清空更稳
    actual_encoding = encoding or {}
    if engine == "scipy":
        kwargs["format"] = "NETCDF3_64BIT"
        actual_encoding = {}

    # 先载入内存，减少懒加载/文件句柄导致的写出冲突
    ds_to_write = ds.load()

    # 显式指定 mode="w"
    ds_to_write.to_netcdf(str(path), mode="w", encoding=actual_encoding, **kwargs)
    return str(path)