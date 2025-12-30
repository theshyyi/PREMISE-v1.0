#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CHIRPS v3.0 Satellite Daily GeoTIFF -> Monthly NetCDF (via PREMISE.geotiff)
===========================================================================

Original script functionality remains unchanged:
- Assumed directory structure: IN_ROOT/2000/chirps-v3.0.sat.2000.01.01.tif ...;
- Assembles daily-scale TIFFs into NetCDF files by year and month:
    OUT_ROOT/2000/pr_daily_2000_01.nc
    OUT_ROOT/2000/pr_daily_2000_02.nc
    ...
- Variable name is pr, unit mm/day, missing values -9999

The core logic is now provided by PREMISE.geotiff.
"""

from pathlib import Path

from premise.geotiff import convert_daily_geotiff_to_monthly_nc

# ===================== config =====================
IN_ROOT  = Path("/home/ud202380664/NPJ_Manuscript/CHIRPSV3.0/CHIRPS_v3_sat_daily")
OUT_ROOT = Path("/home/ud202380664/NPJ_Manuscript/CHIRPSV3.0/CHIRPS_v3_sat_daily_nc_monthly")

YEAR_BEG = 2000
YEAR_END = 2022

VAR_NAME = "pr"
UNITS    = "mm/day"
NODATA   = -9999.0
COMP_LEVEL = 4

# 文件名示例：
#   chirps-v3.0.sat.2000.01.01.tif
# 正则表达式需要有 3 个捕获组 (YYYY, MM, DD)
REGEX_PATTERN = r"chirps-v3\.0\.sat\.(\d{4})\.(\d{2})\.(\d{2})\.tif$"
GLOB_PATTERN  = "chirps-v3.0.sat.*.tif"
# =================================================


def main():
    convert_daily_geotiff_to_monthly_nc(
        in_root=IN_ROOT,
        out_root=OUT_ROOT,
        year_beg=YEAR_BEG,
        year_end=YEAR_END,
        regex_pattern=REGEX_PATTERN,
        glob_pattern=GLOB_PATTERN,
        var_name=VAR_NAME,
        units=UNITS,
        nodata=NODATA,
        comp_level=COMP_LEVEL,
        product_long_name="CHIRPS v3.0 satellite daily precipitation",
        product_source="CHIRPS v3.0 (satellite)",
    )


if __name__ == "__main__":
    main()
