# -*- coding: utf-8 -*-
"""
PREMISE v1.0
============

PREMISE (PREcipitation Multi-source Indices and Software Evaluation)

- 多源降水数据处理
- 干旱 / 极端指数（SPI, SPEI, SRI, STI, ETCCDI-like）
- 网格与网格、站点与网格的多指标评估
- 分区聚合与绘图
- 原始二进制栅格数据 -> NetCDF 转换
"""

from . import io
from . import preprocess
from . import indices
from . import extreme_indices
from . import metrics
from . import evaluation
from . import plotting
from . import pointpixel
from . import workflows
from .reader import binaryio, geotiff, grid, hdf
from . import fusion


__all__ = [
    "io",
    "preprocess",
    "indices",
    "extreme_indices",
    "metrics",
    "evaluation",
    "plotting",
    "pointpixel",
    "workflows",
    "binaryio",
    "geotiff",
    "grid",
    "hdf"
]
