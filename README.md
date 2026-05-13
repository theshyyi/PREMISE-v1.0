# PREMISE v2.1

**PREMISE** (**PRE**cipitation **M**ulti-source **I**ndices and **S**oftware **E**valuation) is an open-source Python package for multi-source precipitation and hydroclimatic data processing. It provides reusable modules for data acquisition planning, format conversion, harmonization, basin-oriented preprocessing, hydroclimatic index calculation, product evaluation, product ranking, and visualization.

This repository is prepared as a functional software repository for journal submission. It includes installation metadata, executable examples based on synthetic data, a regression test suite, and documentation describing data requirements and manuscript workflow reproduction.

## Main capabilities

1. **Data acquisition planning** for common gridded precipitation and hydroclimatic products.
2. **Data conversion and harmonization** from heterogeneous source formats toward standardized NetCDF workflows.
3. **Basin-oriented preprocessing** including spatial, temporal, and variable-level operations.
4. **Hydroclimatic and extreme-index calculation** including SPI, SPEI, SRI, STI, and precipitation-extreme indicators.
5. **Product evaluation** using continuous, event-detection, and spatial diagnostics.
6. **Product ranking and decision support** using single-metric, multi-metric, and consensus ranking methods.
7. **Visualization and reporting** for diagnostic figures and structured summaries.

## Repository structure

```text
PREMISE-v2.1/
├── src/premise/                 # Installable Python package
│   ├── acquisition/             # Source catalogues and acquisition planning
│   ├── conversion/              # Format conversion and harmonization helpers
│   ├── basin/                   # Basin-oriented preprocessing utilities
│   ├── climate_indices/         # Hydroclimatic and extreme-index modules
│   ├── product_evaluation/      # Product benchmarking and metric computation
│   ├── product_ranking/         # Multi-criteria ranking and product selection
│   └── visualization/           # Plotting and reporting utilities
├── examples/                    # Executable synthetic examples
├── tests/                       # Regression and smoke tests
├── docs/                        # User and manuscript-reproduction documentation
├── pyproject.toml               # Package metadata and dependencies
├── CITATION.cff                 # Machine-readable citation metadata
└── LICENSE                      # MIT License
```

## Installation

### Option 1: editable installation for users and reviewers

```bash
git clone https://github.com/theshyyi/PREMISE-v2.1.git
cd PREMISE-v2.1
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Option 2: install optional geospatial and format dependencies

```bash
pip install -e ".[geo]"      # geopandas, regionmask, shapely
pip install -e ".[tiff]"     # rioxarray, rasterio
pip install -e ".[grib]"     # cfgrib, eccodes
pip install -e ".[hdf]"      # h5py, h5netcdf
pip install -e ".[fusion]"   # scikit-learn, joblib, tqdm
pip install -e ".[full]"     # all optional dependency groups
```

## Quick functional check

After installation, the following commands should run without requiring external data:

```bash
python -c "import premise; print(premise.__version__)"
python examples/quickstart/basic_usage.py
python examples/quickstart/module_overview.py
pytest -q
```

Expected result:

```text
2.1.0
SPI-3 computed with shape: ...
RMSE(pr, pr) = 0.0
all tests passed
```

## Minimal Python example

```python
import numpy as np
import pandas as pd
import xarray as xr

from premise.indices import calc_spi
from premise.metrics import rmse

rng = np.random.RandomState(42)
time = pd.date_range("2001-01-01", periods=24, freq="MS")
pr = xr.DataArray(
    rng.gamma(shape=2.0, scale=2.0, size=(24, 2, 2)),
    coords={"time": time, "lat": [30.0, 31.0], "lon": [110.0, 111.0]},
    dims=("time", "lat", "lon"),
    name="pr",
)

spi3 = calc_spi(pr, scale=3, is_monthly_input=True)
print(spi3.name, spi3.shape)
print(rmse(pr, pr))
```

## Executable examples

The examples use synthetic data so that reviewers can run them without downloading large precipitation products.

| Example | Command | Purpose |
|---|---|---|
| Quick start | `python examples/quickstart/basic_usage.py` | Compute SPI and a simple metric from synthetic data |
| Module overview | `python examples/quickstart/module_overview.py` | Show the six-module architecture and source catalogues |
| Basin preprocessing | `python examples/basin_forcing_preparation/run_synthetic_basin_preprocessing.py` | Demonstrate clipping, variable renaming, and preprocessing concepts |
| Product comparison | `python examples/product_comparison/run_synthetic_product_evaluation.py` | Evaluate a synthetic product against a reference |
| Extreme and drought indices | `python examples/extreme_and_drought_indices/run_synthetic_indices.py` | Compute SPI and selected precipitation-extreme indices |
| Product ranking | `python examples/product_ranking/run_synthetic_ranking.py` | Rank synthetic products using multi-criteria metrics |

## Documentation

- [Installation](docs/installation.md)
- [Quick start](docs/quickstart.md)
- [Input data requirements](docs/data_requirements.md)
- [Module architecture](docs/module_architecture.md)
- [API reference overview](docs/api_reference.md)
- [Examples](docs/examples.md)
- [Reproducing manuscript workflows](docs/reproduce_manuscript.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Software availability](docs/software_availability.md)

## Testing

Run the regression tests with:

```bash
pytest -q
```

The test suite checks package importability, core metrics, SPI calculation, architecture-level helper functions, and backward-compatible metadata parsing.

## Input data requirements

For real applications, PREMISE expects analysis-ready or convertible gridded datasets with explicit coordinate and metadata conventions:

- dimensions: usually `time`, `lat`, `lon` or equivalent names that can be standardized;
- coordinates: latitude and longitude in geographic coordinates unless reprojected explicitly;
- precipitation units: preferably `mm day-1` for daily rates or `mm` for accumulated totals;
- time coordinate: CF-decodable datetime values;
- variables: user-specified or inferable precipitation, temperature, runoff, soil moisture, or PET variables;
- missing values: encoded as NaN or documented missing-value attributes.

More detail is provided in [docs/data_requirements.md](docs/data_requirements.md).

## Citation

Please cite the software release and associated manuscript when using PREMISE. A machine-readable citation file is provided in [CITATION.cff](CITATION.cff). The repository should also be archived as a versioned release on Zenodo for journal submission.

## License

PREMISE is released under the MIT License. See [LICENSE](LICENSE).
