#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute SPI/SPEI/SRI/STI and regional means for a given dataset.

Example usage:

    python scripts/run_indices_by_region.py \
        --nc /home/.../pr_tas_pet_runoff.nc \
        --shp /home/.../china_7regions.shp \
        --region-field climate \
        --out-nc /home/.../indices_regions.nc
"""

import argparse

import xarray as xr

from premise.workflows import compute_indices_and_regional_means


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute hydroclimatic indices and regional means."
    )
    parser.add_argument("--nc", required=True, help="Input NetCDF file.")
    parser.add_argument("--shp", required=True, help="Shapefile defining regions.")
    parser.add_argument(
        "--region-field",
        default=None,
        help="Field name in shapefile used as region labels (e.g., 'climate').",
    )
    parser.add_argument(
        "--out-nc",
        required=True,
        help="Output NetCDF file containing indices and regional means.",
    )
    parser.add_argument(
        "--spi-scales",
        default="3,6,12",
        help="Comma-separated SPI scales in months (e.g., '3,6,12').",
    )
    parser.add_argument(
        "--spei-scales",
        default="3,6,12",
        help="Comma-separated SPEI scales in months.",
    )
    parser.add_argument(
        "--sri-scales",
        default="",
        help="Comma-separated SRI scales in months (optional).",
    )
    parser.add_argument(
        "--sti-scales",
        default="",
        help="Comma-separated STI scales in months (optional).",
    )
    parser.add_argument(
        "--precip-var",
        default="pr",
        help="Precipitation variable name in NetCDF.",
    )
    parser.add_argument(
        "--temp-var",
        default=None,
        help="Temperature variable name in NetCDF (for STI).",
    )
    parser.add_argument(
        "--pet-var",
        default=None,
        help="PET variable name in NetCDF (for SPEI).",
    )
    parser.add_argument(
        "--runoff-var",
        default=None,
        help="Runoff variable name in NetCDF (for SRI).",
    )
    return parser.parse_args()


def _parse_int_list(s: str):
    s = s.strip()
    if not s:
        return []
    return [int(v) for v in s.split(",") if v.strip()]


def main():
    args = parse_args()

    ds = xr.open_dataset(args.nc)

    spi_scales = _parse_int_list(args.spi_scales)
    spei_scales = _parse_int_list(args.spei_scales)
    sri_scales = _parse_int_list(args.sri_scales)
    sti_scales = _parse_int_list(args.sti_scales)

    out_ds = compute_indices_and_regional_means(
        ds,
        precip_var=args.precip_var,
        temp_var=args.temp_var,
        pet_var=args.pet_var,
        runoff_var=args.runoff_var,
        shp_path=args.shp,
        region_field=args.region_field,
        spi_scales=spi_scales,
        spei_scales=spei_scales,
        sri_scales=sri_scales,
        sti_scales=sti_scales,
    )

    out_ds.to_netcdf(args.out_nc)
    print(f"[premise] Indices and regional means saved to: {args.out_nc}")


if __name__ == "__main__":
    main()
