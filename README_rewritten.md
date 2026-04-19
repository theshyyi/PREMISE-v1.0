# PREMISE v2.0

**PREMISE** (**PRE**cipitation **M**ulti-source **I**ndices and **S**oftware **E**valuation) is an open-source Python framework for **multi-source hydroclimatic data preparation, basin-oriented preprocessing, product evaluation, climate-index generation, product ranking, and visualization**. It is designed to support reproducible workflows for gridded hydroclimatic datasets used in environmental modelling, hydrology, and climate-diagnostic studies.

The current manuscript-oriented release is organized around **seven core modules**:

1. **Data acquisition**
2. **Data conversion and harmonization**
3. **Basin-oriented preprocessing**
4. **Product evaluation**
5. **Climate indices**
6. **Product ranking and fusion**
7. **Visualization and reporting**

This README has been reorganized to match the current modular architecture. The previous README described a six-module layout and still referenced legacy compatibility layers that are no longer part of the cleaned core structure. fileciteturn5file0

---

## 1. What PREMISE is designed for

PREMISE provides an end-to-end workflow for heterogeneous hydroclimatic products, especially precipitation-related datasets, from raw-data ingestion to publication-ready diagnostics. The framework is intended for users who need to:

- download or register multiple hydroclimatic products,
- standardize heterogeneous raw formats into analysis-ready datasets,
- clip and preprocess products for a study basin or region,
- evaluate candidate products against reference data,
- compute climate and drought indices,
- rank products using single-metric or multi-criteria decision approaches,
- build fused products from top-performing candidates, and
- generate consistent figures and reports for manuscripts.

---

## 2. Core architecture

| Module | Package path | Primary role | Typical inputs | Typical outputs |
|---|---|---|---|---|
| Data acquisition | `premise.acquisition` | Register data sources and prepare download workflows | Product definitions, source metadata | Download plans, source templates |
| Data conversion and harmonization | `premise.conversion` | Convert raw files to standardized NetCDF and unify metadata | Binary, HDF/HDF5, GRIB, GeoTIFF, NetCDF | Standardized NetCDF |
| Basin-oriented preprocessing | `premise.basin` | Clip, mask, aggregate, rename, subset, and resample data for a target region | Standardized NetCDF, shapefile/GeoJSON, bounding box | Basin-ready NetCDF and summaries |
| Product evaluation | `premise.product_evaluation` | Compare candidate products against reference datasets | NetCDF products, station tables, reference data | Spatial metrics (NetCDF), summary tables |
| Climate indices | `premise.climate_indices` | Compute extreme climate and hydroclimatic indices | Daily or monthly hydroclimatic variables | Index NetCDF and summary tables |
| Product ranking and fusion | `premise.product_ranking` | Rank products using multiple methods and build fused datasets | Evaluation tables and selected product files | Ranking tables, consensus results, fused NetCDF |
| Visualization and reporting | `premise.visualization` | Produce maps, diagrams, scatter plots, ranking plots, and report graphics | Tables, NetCDF, ranking results, evaluation results | Publication-ready figures and summaries |

Two additional utilities, **correction** and **fusion**, may still exist in the repository as auxiliary or experimental components, but they are not treated as part of the seven-module manuscript core.

---

## 3. Recommended workflow

A typical PREMISE workflow proceeds as follows:

1. **Acquire** or register multiple candidate datasets.
2. **Convert** heterogeneous source files into standardized NetCDF.
3. **Preprocess** the datasets to the study basin, region, or grid.
4. **Evaluate** products against reference data at grid or station scale.
5. **Compute climate indices** where diagnostic analysis is required.
6. **Rank products** using one or more decision strategies.
7. **Fuse top-ranked products** if a composite dataset is desired.
8. **Visualize and report** the outputs for interpretation and publication.

This ordering aligns with the current internal restructuring of the framework, where ranking is treated as a dedicated module placed between evaluation and visualization rather than being merged into evaluation. The previous README did not yet reflect this separation. fileciteturn5file0

---

## 4. Installation

### Basic installation

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

### Typical optional dependencies by module

- `regionmask`, `geopandas`, `shapely` for basin preprocessing
- `rasterio` for GeoTIFF conversion
- `h5py` for HDF/HDF5 conversion
- `cfgrib` and `eccodes` for GRIB conversion
- `cartopy` and `matplotlib` for visualization

---

## 5. Repository-level view

A cleaned manuscript-oriented repository is expected to expose the following main packages under `premise/`:

```text
premise/
├─ acquisition/
├─ conversion/
├─ basin/
├─ product_evaluation/
├─ climate_indices/
├─ product_ranking/
├─ visualization/
├─ correction/          # optional / auxiliary
└─ fusion/              # optional / auxiliary
```

This layout replaces the older top-level pattern in which evaluation, indices, plotting, and preprocessing utilities were also duplicated as standalone root modules. The current README should describe the cleaned modular structure rather than the earlier mixed layout. fileciteturn5file0

---

## 6. Module summaries

### 6.1 `premise.acquisition`

This module manages source registration and download-oriented metadata for hydroclimatic products. It is intended to make heterogeneous external data sources traceable and reproducible before any analysis begins.

Typical use cases include:

- registering product families,
- listing products by variable or category,
- preparing download templates, and
- standardizing product metadata for later pipeline stages.

### 6.2 `premise.conversion`

This module standardizes heterogeneous source formats into NetCDF-based datasets suitable for downstream workflows. It is particularly important for products originally distributed as:

- binary files,
- HDF/HDF5,
- GRIB,
- GeoTIFF stacks.

The conversion layer should also record conversion summaries, timing, metadata harmonization, and output provenance.

### 6.3 `premise.basin`

This module prepares data for a target study region. Supported operations typically include:

- bounding-box clipping,
- shapefile or GeoJSON masking,
- time subsetting,
- variable renaming,
- spatial aggregation, and
- resampling to a target grid or target resolution.

### 6.4 `premise.product_evaluation`

This module focuses only on **product evaluation**, not climate-index generation. It supports comparison against:

- **reference NetCDF datasets** (grid-to-grid evaluation), and
- **observational tables** such as station CSV/Excel files (table-to-grid evaluation).

Typical outputs include:

- spatial metric fields in NetCDF format,
- overall and grouped metrics in CSV/XLSX format,
- station-level summaries,
- monthly or seasonal evaluation tables.

### 6.5 `premise.climate_indices`

This module is dedicated to climate-index generation and is separate from product evaluation. It can be used for:

- extreme precipitation indices,
- drought indices,
- hydroclimatic standardized indices.

Typical outputs are NetCDF fields and summary tables for downstream regional analysis or manuscript figures.

### 6.6 `premise.product_ranking`

This module ranks multiple products of the same type using evaluation outputs. It should support:

- direct ascending/descending single-metric ranking,
- simple multi-metric rank aggregation,
- multi-criteria decision methods such as **TOPSIS**,
- objective weighting approaches such as **CRITIC** and **entropy weighting**,
- multiple-method consensus ranking,
- selection of the top-performing products,
- generation of fused composite products from the top-ranked subset.

Typical outputs include ranking tables, method-comparison tables, consensus results, and fused NetCDF products.

### 6.7 `premise.visualization`

This module provides publication-ready visual expression for different result types. Different outputs should be matched with different figure types, for example:

- **spatial NetCDF results** → maps and spatial panels,
- **evaluation tables** → heatmaps, grouped bar plots, boxplots, violin plots,
- **product-vs-reference diagnostics** → density scatter plots, Taylor diagrams, SAL diagrams,
- **event-detection results** → performance diagrams,
- **ranking results** → ranking bars and ranking heatmaps,
- **time series** → line plots and grouped temporal summaries.

The older README already highlighted several existing figure types such as spatial maps, Taylor diagrams, SAL diagrams, boxplots, violin plots, density scatter plots, and performance diagrams. The new README simply reorganizes them under a dedicated visualization module and ties them to result types more explicitly. fileciteturn5file0turn4file0

---

## 7. Typical outputs by module

| Module | Main outputs |
|---|---|
| Acquisition | source metadata, download templates |
| Conversion | standardized NetCDF, conversion logs, conversion summary tables |
| Basin | clipped/resampled NetCDF, preprocessing reports |
| Product evaluation | metric NetCDF, overall/grouped evaluation tables |
| Climate indices | index NetCDF, index summary tables |
| Product ranking | ranking CSV/XLSX, method comparison tables, fused NetCDF |
| Visualization | figures in PNG/PDF/SVG and figure-level summaries |

---

## 8. Quick start concept

A full manuscript workflow normally begins with heterogeneous source data and ends with ranked, fused, and visualized products. A concise conceptual pattern is:

```python
import premise

# 1. acquisition
# 2. conversion
# 3. basin preprocessing
# 4. product evaluation
# 5. climate indices
# 6. product ranking
# 7. visualization
```

For reproducible project execution, a repository-level master pipeline script is recommended to orchestrate these stages in sequence.

---

## 9. Reproducibility and reporting

For manuscript-oriented use, each module should ideally generate:

- machine-readable outputs (`nc`, `csv`, `json`),
- human-readable summaries (`md`, report tables),
- timing and processing logs,
- explicit task configuration records.

This makes it easier to cite:

- what data were used,
- how they were converted,
- how they were clipped and aligned,
- how products were evaluated and ranked,
- how figures were produced.

---

## 10. Software availability

- **Name of software:** PREMISE v1.0
- **Description:** An open-source Python framework for hydroclimatic data acquisition, conversion, basin-oriented preprocessing, product evaluation, climate-index generation, product ranking, and visualization.
- **Developer:** Xinlong Le
- **Programming language:** Python
- **System requirements:** Linux, Windows, or macOS
- **Repository:** https://github.com/theshyyi/PREMISE-v1.0
- **Version:** 1.0.0
- **DOI:** 10.5281/zenodo.18093220

---

## 11. Testing

If tests are included in the repository, they can be run with:

```bash
pytest -q
```

For large workflow modules, dedicated real-data runner scripts are recommended in addition to unit tests.

---

## 12. Citation and license

Please cite the associated Zenodo archive when using PREMISE in research outputs. A machine-readable citation file may be provided as `CITATION.cff`.

This project is released under the MIT License. See `LICENSE` for details.
