from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import xarray as xr


def _open_variable(path: str | Path, var_name: str | None = None) -> xr.DataArray:
    ds = xr.open_dataset(path)
    try:
        if var_name is None:
            if len(ds.data_vars) != 1:
                raise ValueError(f'Please specify var_name for {path}; found variables: {list(ds.data_vars)}')
            var_name = list(ds.data_vars)[0]
        da = ds[var_name].load()
    finally:
        ds.close()
    return da


def _align_arrays(arrays: list[xr.DataArray]) -> list[xr.DataArray]:
    aligned = xr.align(*arrays, join='inner')
    return list(aligned)


def fuse_top_products_mean(
    product_paths: Mapping[str, str | Path],
    top_products: Sequence[str],
    *,
    var_name: str | None = None,
    output_path: str | Path | None = None,
    fused_name: str = 'fused_product',
) -> xr.Dataset:
    arrays = [_open_variable(product_paths[p], var_name) for p in top_products]
    arrays = _align_arrays(arrays)
    stack = xr.concat(arrays, dim='product')
    stack = stack.assign_coords(product=('product', list(top_products)))
    fused = stack.mean(dim='product', skipna=True).rename(fused_name)
    ds_out = fused.to_dataset(name=fused_name)
    ds_out.attrs['source_products'] = ','.join(map(str, top_products))
    ds_out.attrs['fusion_method'] = 'simple_mean'
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(output_path)
    return ds_out


def fuse_top_products_weighted(
    product_paths: Mapping[str, str | Path],
    product_weights: Mapping[str, float],
    *,
    var_name: str | None = None,
    output_path: str | Path | None = None,
    fused_name: str = 'fused_product',
) -> xr.Dataset:
    prods = list(product_weights.keys())
    weights = np.asarray([float(product_weights[p]) for p in prods], dtype=float)
    if np.sum(weights) <= 0:
        raise ValueError('product_weights must sum to a positive value')
    weights = weights / np.sum(weights)
    arrays = [_open_variable(product_paths[p], var_name) for p in prods]
    arrays = _align_arrays(arrays)
    stack = xr.concat(arrays, dim='product')
    stack = stack.assign_coords(product=('product', prods))
    w = xr.DataArray(weights, dims=('product',), coords={'product': prods})
    fused = (stack * w).sum(dim='product', skipna=True).rename(fused_name)
    ds_out = fused.to_dataset(name=fused_name)
    ds_out.attrs['source_products'] = ','.join(map(str, prods))
    ds_out.attrs['source_weights'] = ','.join([f'{p}:{float(product_weights[p]):.6f}' for p in prods])
    ds_out.attrs['fusion_method'] = 'weighted_mean'
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(output_path)
    return ds_out
