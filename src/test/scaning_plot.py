#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
from matplotlib import rcParams

from premise.plotting import multi_scatter_density_products

rcParams.update({
    "font.family": ["Times New Roman"],
    "font.size": 14,
    "mathtext.fontset": "stix",
})

# ===== 路径配置 =====
REF_FILE = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/CMFDV2.TIMEFIX.daily.CHINA.nc"
PROD_FILES = {
    "GSMAP_MVK": "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/GSMAP_MVK.TIMEFIX.daily.CHINA.nc",
    "CHIRPSv3":  "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/CHIRPSv3.TIMEFIX.daily.CHINA.nc",
    "IMERG":     "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/IMERG_V06.TIMEFIX.daily.CHINA.nc",
    "MSWEP":     "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/MSWEP_V280.TIMEFIX.daily.CHINA.nc",
}

VAR_NAME   = "pr"
TIME_START = "2000-01-01"
TIME_END   = "2020-12-31"
OUT_PNG    = "/home/ud202380664/PRE_MERGE/Figures/scatter_clim_4products_vs_CMFDV2.png"


def main():
    # 参考产品 climatology
    ref = xr.open_dataset(REF_FILE)[VAR_NAME].sel(time=slice(TIME_START, TIME_END))
    ref_clim = ref.mean(dim="time", skipna=True)

    # 各产品 climatology
    tests_clim = {}
    for name, path in PROD_FILES.items():
        da = xr.open_dataset(path)[VAR_NAME].sel(time=slice(TIME_START, TIME_END))
        # 与参考产品在空间上对齐
        da_align, ref_align = xr.align(da, ref, join="inner")
        # 注意：ref_clim 也可以用 ref_align 再算一遍，这里直接用 ref_clim 对齐后的网格
        tests_clim[name] = da_align.mean(dim="time", skipna=True)

    # 调用 PREMISE 的多面板绘图
    fig, axes, stats_all = multi_scatter_density_products(
        ref=ref_clim,
        tests=tests_clim,
        x_label="CMFD-V2 climatology (mm)",
        y_label="Product climatology (mm)",
        title="Climatological rainfall: Products vs CMFD-V2",
        out_path=OUT_PNG,
        cmap="gist_rainbow",
        point_size=3.0,
        tick_step=5.0,
        ncols=2,   # 4 个产品 -> 2×2 布局
    )

    print("各产品统计量：")
    for name, st in stats_all.items():
        print(name, st)


if __name__ == "__main__":
    main()
