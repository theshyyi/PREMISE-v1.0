# API reference overview

This document lists the main public entry points used in examples and tests.

## Core package

```python
import premise
print(premise.__version__)
```

## Metrics

```python
from premise.metrics import bias, mae, rmse, corr, kge, nse, pod, far, csi
```

## Hydroclimatic indices

```python
from premise.indices import calc_spi, calc_spei, calc_sri, calc_sti
from premise.climate_indices import rx1day, rx5day, prcptot, cdd, cwd
```

## Product evaluation

```python
from premise.product_evaluation import evaluate_grid_pair, evaluate_grid_by_group, compute_spatial_metric_dataset
```

## Product ranking

```python
from premise.product_ranking import average_rank, topsis_rank, weighted_sum_rank, consensus_rank
```

## Conversion and harmonization

```python
from premise.conversion import summarize_harmonization_scope, detect_format, open_standard_dataset
```

Optional functions may require optional dependencies such as `geopandas`, `rasterio`, `cfgrib`, or `h5py`.
