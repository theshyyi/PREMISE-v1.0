# Reproducing manuscript workflows

This repository provides lightweight synthetic examples that reproduce the software workflows described in the manuscript without requiring large external datasets.

## Workflow 1: Data harmonization and source catalogue

Command:

```bash
python examples/quickstart/module_overview.py
```

Expected result: printed source catalogue entries, harmonization scope, and evaluation-to-ranking workflow description.

## Workflow 2: Hydroclimatic index calculation

Command:

```bash
python examples/extreme_and_drought_indices/run_synthetic_indices.py
```

Expected result: SPI and precipitation-extreme outputs generated from a synthetic precipitation dataset.

## Workflow 3: Product evaluation

Command:

```bash
python examples/product_comparison/run_synthetic_product_evaluation.py
```

Expected result: a metrics table comparing a synthetic candidate product against a synthetic reference.

## Workflow 4: Product ranking

Command:

```bash
python examples/product_ranking/run_synthetic_ranking.py
```

Expected result: a ranked product table using multi-criteria diagnostic metrics.

## Workflow 5: Basin-oriented preprocessing

Command:

```bash
python examples/basin_forcing_preparation/run_synthetic_basin_preprocessing.py
```

Expected result: a clipped and renamed synthetic gridded precipitation object.

## Full technical check

```bash
python -c "import premise; print(premise.__version__)"
python examples/quickstart/basic_usage.py
python examples/quickstart/module_overview.py
pytest -q
```
