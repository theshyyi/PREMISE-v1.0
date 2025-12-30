# PREMISE v1.0

PREMISE (PREcipitation Multi-source Indices and Software Evaluation) is an open-source Python framework for:

- Multi-source precipitation data processing
- Extreme and drought index calculation (SPI, SPEI, SRI, STI, etc.)
- Product evaluation with continuous and detection-based metrics
- Regional aggregation and diagnostic plotting

## Installation

```bash
pip install -e .
import xarray as xr
from premise.indices import calc_spi
from premise.preprocess import area_mean_by_region
from premise.plotting import plot_violin_groups

ds = xr.open_dataset("pr_daily.nc")
spi3 = calc_spi(ds["pr"], scale=3)

spi3_reg = area_mean_by_region(
    spi3,
    "china_7regions.shp",
    region_field="climate",
)

data_dict = {
    str(r): spi3_reg.sel(region=r).values
    for r in spi3_reg.region.values
}

fig, ax = plot_violin_groups(
    data_dict,
    title="SPI-3 by climate region",
    ylabel="SPI-3"
)
fig.savefig("spi3_violin_regions.png", dpi=300)

---

## 4. `src/premise/__init__.py`

```python
# -*- coding: utf-8 -*-
"""
PREMISE v1.0
============

PREMISE (PREcipitation Multi-source Indices and Software Evaluation) is an
open-source Python framework for:

- Multi-source precipitation data processing
- Extreme and drought index calculation (SPI, SPEI, SRI, STI)
- Product evaluation (BIAS, RMSE, KGE, POD, FAR, CSI, FBIAS, etc.)
- Regional aggregation and diagnostic plotting
- Workflow-oriented grid and station (point–pixel) analysis
"""

from . import io
from . import preprocess
from . import indices
from . import metrics
from . import plotting
from . import pointpixel
from . import workflows

__all__ = [
    "io",
    "preprocess",
    "indices",
    "metrics",
    "plotting",
    "pointpixel",
    "workflows",
]
