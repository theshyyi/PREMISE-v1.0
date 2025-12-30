#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
from matplotlib import rcParams

from premise.plotting import scatter_density_product

# ========== 全局字体（按你习惯） ==========
rcParams.update({
    "font.family": ["Times New Roman"],
    "font.size": 18,
    "mathtext.fontset": "stix",
})

# ========== 基本配置 ==========
REF_FILE  = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/CMFDV2.TIMEFIX.daily.CHINA.nc"
TEST_FILE = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish/GSMAP_MVK.TIMEFIX.daily.CHINA.nc"

VAR_NAME   = "pr"
TIME_START = "2000-01-01"
TIME_END   = "2020-12-31"      # 根据你的研究时段调整
OUT_PNG    = "/home/ud202380664/PRE_MERGE/Figures/scatter_clim_GSMAP_MVK_CMFDV2.png"


def main():
    # 1. 打开参考和待评估产品
    ds_ref = xr.open_dataset(REF_FILE)[VAR_NAME].sel(time=slice(TIME_START, TIME_END))
    ds_test = xr.open_dataset(TEST_FILE)[VAR_NAME].sel(time=slice(TIME_START, TIME_END))

    # 2. 计算多年平均气候态（每个网格点一个值）
    ref_clim = ds_ref.mean(dim="time", skipna=True)
    test_clim = ds_test.mean(dim="time", skipna=True)

    # 3. 对齐网格（如果两个数据在空间上已经完全一致，可以省略这一步）
    ref_clim_aligned, test_clim_aligned = xr.align(ref_clim, test_clim, join="inner")

    # 4. 调用 PREMISE 的绘图函数
    title = "Climatological rainfall (GSMAP_MVK vs CMFD-V2)"
    x_label = "CMFD-V2 climatology (mm)"
    y_label = "GSMAP_MVK climatology (mm)"

    fig, ax, stats = scatter_density_product(
        ref_clim_aligned,
        test_clim_aligned,
        x_label=x_label,
        y_label=y_label,
        title=title,
        out_path=OUT_PNG,
        cmap="gist_rainbow",
        point_size=4.0,
        tick_step=5.0,
    )

    print("统计量：", stats)
    # 如需交互查看：
    # import matplotlib.pyplot as plt
    # plt.show()


if __name__ == "__main__":
    main()
