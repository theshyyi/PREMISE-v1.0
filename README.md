# PREMISE v2.0

**PREMISE** (PREcipitation Multi-source Indices and Software Evaluation) is an open-source Python framework for **basin-oriented hydroclimatic data preparation, comparative product evaluation, and climate-diagnostic workflows** in environmental modelling applications.

The redesigned v1.0 architecture is organized around six application-oriented modules:

1. **Data acquisition** for multi-source hydroclimatic products.
2. **Data conversion and harmonization** for heterogeneous raw formats and metadata.
3. **Basin-oriented preprocessing** for clipping, masking, aggregation, and forcing preparation.
4. **Product evaluation** for benchmarking multi-source products against a reference dataset.
5. **Extreme climate and drought indices** for precipitation and hydroclimatic diagnostics.
6. **Visualization and reporting** for publication-ready figures and structured summaries.

Optional correction and fusion utilities are retained as **legacy or experimental extensions**. They are not part of the six-module core architecture described in the current manuscript revision.

---

## Homepage module graphic placeholder

The repository root already includes a dedicated asset path for the module icon board you plan to add later.

<p align="center">
  <img src="assets/module-icons/overview-placeholder.svg" alt="PREMISE module icon board placeholder" width="900">
</p>

When the final graphic is ready, replace the file at `assets/module-icons/overview-placeholder.svg` or update the image path in this section.

---

## Software availability

- **Name of software:** PREMISE v2.0
- **Description:** An open-source Python framework for basin-oriented preprocessing, harmonization, index generation, and comparative evaluation of multi-source gridded hydroclimatic datasets in environmental modelling workflows.
- **Developer:** Xinlong Le
- **Programming language:** Python
- **System requirements:** Linux, Windows, or macOS
- **Repository:** https://github.com/theshyyi/PREMISE-v1.0
- **Version:** 1.0.0
- **DOI:** 10.5281/zenodo.18093220

A manuscript-ready note is provided in [`docs/software_availability.md`](docs/software_availability.md).

---

## Installation

### Core installation

```bash
pip install -e .
```

### Optional extras

```bash
pip install -e .[geo]
pip install -e .[tiff]
pip install -e .[grib]
pip install -e .[hdf]
pip install -e .[fusion]
pip install -e .[full]
```

---

## Six-module architecture

| Module | Package path | Primary role |
|---|---|---|
| Data acquisition | `premise.acquisition` | Register and query hydroclimatic data sources, build download plans |
| Data conversion and harmonization | `premise.conversion` | Convert heterogeneous raw files to standardized NetCDF-ready datasets |
| Basin-oriented preprocessing | `premise.basin` | Clip, mask, aggregate, and prepare basin-facing forcing products |
| Product evaluation | `premise.product_evaluation` | Evaluate candidate products against a reference and summarize ranking workflows |
| Extreme climate and drought indices | `premise.climate_indices` | Compute drought indices and precipitation-extreme indicators |
| Visualization and reporting | `premise.visualization` | Generate maps, performance diagrams, Taylor diagrams, and grouped summaries |

Legacy modules such as `premise.preprocess`, `premise.indices`, `premise.evaluation`, and `premise.plotting` are still available for backward compatibility.

---

## Quick start with the new module layout

### 1. Explore hydroclimatic source templates

```python
from premise.acquisition import list_sources

for source in list_sources(variable="precipitation"):
    print(source.key, source.title, source.temporal_resolution)
```

### 2. Convert heterogeneous data to standardized NetCDF

```python
from premise.conversion import convert_binary_to_nc

convert_binary_to_nc(
    meta_path="cpc_global_daily.meta",
    data_path="precip_20000101.bin",
    out_nc="precip_20000101.nc",
)
```

### 3. Prepare basin-oriented forcing summaries

```python
from premise.basin import area_mean_by_region
from premise.climate_indices import calc_spi
import xarray as xr

pr = xr.open_dataset("precip_20000101.nc")["pr"]
spi3 = calc_spi(pr, scale=3)
regional = area_mean_by_region(spi3, "regions.shp")
print(regional)
```

### 4. Summarize evaluation or ranking workflow stages

```python
from premise.product_evaluation import describe_ranking_workflow

print(describe_ranking_workflow())
```

---

## Examples

- [`examples/quickstart/basic_usage.py`](examples/quickstart/basic_usage.py) keeps the original minimal workflow.
- [`examples/quickstart/module_overview.py`](examples/quickstart/module_overview.py) demonstrates the redesigned module structure.
- [`examples/basin_forcing_preparation/`](examples/basin_forcing_preparation) is reserved for a basin-facing preprocessing case.
- [`examples/product_comparison/`](examples/product_comparison) is reserved for a comparative benchmarking case.
- [`examples/extreme_and_drought_indices/`](examples/extreme_and_drought_indices) is reserved for an index-generation case.

---

## Documentation

- [`docs/software_availability.md`](docs/software_availability.md)
- [`docs/benchmark_template.md`](docs/benchmark_template.md)
- [`docs/module_architecture.md`](docs/module_architecture.md)
- [`docs/github_homepage_branding.md`](docs/github_homepage_branding.md)

---

## Testing

Minimal regression-oriented tests are provided under [`tests/`](tests):

```bash
pytest -q
```

---

## Citation and license

Please cite the associated Zenodo archive when using PREMISE in research outputs. A machine-readable citation file is included as [`CITATION.cff`](CITATION.cff).

This project is released under the MIT License. See [`LICENSE`](LICENSE).
