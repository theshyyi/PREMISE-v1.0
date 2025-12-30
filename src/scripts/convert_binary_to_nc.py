#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
convert_binary_to_nc.py
=======================

命令行工具：基于 PREMISE.binaryio，将一个二进制栅格文件转换为 NetCDF。

示例：
    python convert_binary_to_nc.py \
        --meta /path/to/meta_pr.txt \
        --data /path/to/file_20000101.bin.gz \
        --out  /path/to/file_20000101.nc

如果 meta 文件里已经写了 data_path，可以不传 --data 参数。
"""

import argparse

from premise.binaryio import convert_binary_to_netcdf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw binary grid file (bin/dat/bin.gz) to NetCDF using PREMISE meta txt."
    )
    parser.add_argument("--meta", required=True, help="Path to meta txt file.")
    parser.add_argument(
        "--data",
        required=False,
        default=None,
        help="Path to binary data file. If omitted, use data_path in meta.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output NetCDF path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_binary_to_netcdf(
        meta_path=args.meta,
        out_nc=args.out,
        data_path=args.data,
    )


if __name__ == "__main__":
    main()
