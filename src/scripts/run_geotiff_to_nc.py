#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CLI wrapper for daily GeoTIFF to monthly NetCDF conversion."""

import argparse
from pathlib import Path

from premise.geotiff import convert_daily_geotiff_to_monthly_nc


def parse_args():
    ap = argparse.ArgumentParser(description="Convert daily GeoTIFF files to monthly NetCDF outputs.")
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--year-beg", type=int, required=True)
    ap.add_argument("--year-end", type=int, required=True)
    ap.add_argument("--var-name", default="pr")
    ap.add_argument("--units", default="mm/day")
    ap.add_argument("--nodata", type=float, default=-9999.0)
    ap.add_argument("--comp-level", type=int, default=4)
    ap.add_argument("--regex-pattern", default=r"chirps-v3\.0\.sat\.(\d{4})\.(\d{2})\.(\d{2})\.tif$")
    ap.add_argument("--glob-pattern", default="chirps-v3.0.sat.*.tif")
    return ap.parse_args()


def main():
    args = parse_args()
    convert_daily_geotiff_to_monthly_nc(
        in_root=Path(args.in_root),
        out_root=Path(args.out_root),
        year_beg=args.year_beg,
        year_end=args.year_end,
        regex_pattern=args.regex_pattern,
        glob_pattern=args.glob_pattern,
        var_name=args.var_name,
        units=args.units,
        nodata=args.nodata,
        comp_level=args.comp_level,
    )


if __name__ == "__main__":
    main()
