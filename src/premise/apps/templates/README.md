# PREMISE application-layer templates

These templates provide user-facing entry points on top of the existing low-level modules.

## Apps

1. `hydro_data_builder`
   - download
   - convert to NetCDF
   - clip by shapefile or bbox
   - clip by time
   - rename variables
   - resample to target grid
   - merge all prepared outputs into a single hydrological-model input dataset

2. `dataset_evaluator`
   - grid-to-grid evaluation
   - grid-to-station evaluation
   - batch metric computation
   - summary table export
   - summary heatmap export

3. `product_ranker`
   - multi-metric product ranking
   - top-N selection
   - ranking bar plot
   - ranking-score heatmap

## Unified entry

```python
from premise.apps import run_application_from_file
run_application_from_file("path/to/config.json")
```

## Notes

The application layer is intentionally thin. It reuses the existing modules:
- `premise.acquisition`
- `premise.conversion`
- `premise.basin`
- `premise.product_evaluation`
- `premise.product_ranking`
- `premise.visualization`

It is a task-organizing layer rather than a replacement of the core modules.
