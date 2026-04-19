from __future__ import annotations

from matplotlib import rcParams


def setup_mpl_fonts(use_chinese: bool = True, base_size: int = 10) -> None:
    rcParams["font.family"] = "sans-serif"
    if use_chinese:
        rcParams["font.sans-serif"] = ["Times New Roman", "SimHei", "Arial Unicode MS"]
    else:
        rcParams["font.sans-serif"] = ["Times New Roman"]
    rcParams["axes.titlesize"] = base_size + 1
    rcParams["axes.labelsize"] = base_size
    rcParams["xtick.labelsize"] = base_size - 1
    rcParams["ytick.labelsize"] = base_size - 1
    rcParams["legend.fontsize"] = base_size - 1
    rcParams["figure.dpi"] = 150
    rcParams["savefig.dpi"] = 300
