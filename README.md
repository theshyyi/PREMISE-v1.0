# PREMISE v1.0

**PREMISE** (PREcipitation Multi-source Indices and Software Evaluation) is an open-source, Python-first framework designed to make **multi-source precipitation data** easier to **standardize, evaluate, correct, and fuse** for hydro‑climate applications.

The current v1.0 release targets **China-scale** precipitation evaluation and regional diagnostics (e.g., 7 climatic regions), while remaining general enough to be applied to any user-provided region masks or shapefiles.

---

## Key capabilities

### Data I/O and standardization
- Read multi-format precipitation datasets through a unified interface (xarray-based).
- Convert non-NetCDF sources to NetCDF when a backend is available:
  - **Raw binary rasters** (e.g., `.bin`, `.dat`, `.gz`) via `premise.binaryio`.
  - **Daily GeoTIFF stacks** (e.g., CHIRPS daily TIFF) to monthly NetCDF via `premise.geotiff`.
  - **GRIB / HDF** can be supported through xarray backends (e.g., `cfgrib`, `h5netcdf/h5py`) when installed.

### Indices (drought and extremes)
- Drought indices: **SPI, SPEI, SRI, STI** (multi-scale).
- Extreme climate indices (ETCCDI-style): precipitation intensity, frequency, persistence, and percentile-based indicators (module: `premise.extreme_indices`).

### Evaluation and diagnostics
- Continuous metrics: Bias, RMSE/MAE, correlation, NSE, KGE, etc.
- Event/detection metrics: POD, FAR, CSI, HSS, frequency bias, etc.
- Regional aggregation for user-defined regions (e.g., China climatic regions, basins).
- Publication-oriented diagnostic plots (maps and distribution plots).

### Point–pixel and workflow utilities
- Station-to-grid (point–pixel) utilities for evaluation workflows.
- Scriptable batch workflows for grid evaluation and regional index calculation (see `scripts/`).

### (Optional/extended) correction and fusion
- Two-stage Random Forest precipitation correction with wet/dry classification + residual regression and monthly scaling constraints.
- Regional-aware fusion toolbox (baselines + benchmarks + RF two-stage fusion), designed for ablation studies and reproducible experiments.

> Note: Correction and fusion modules are often installed with extra optional dependencies (scikit-learn, joblib, geopandas, regionmask).

---

## Installation

### Editable install (recommended during development)
```bash
pip install -e .
```

### Optional dependencies
Depending on which features you use, you may need additional backends:

- GeoTIFF conversion: `rasterio` (and optionally `rioxarray`)
- GRIB reading: `cfgrib` + ECMWF eccodes
- HDF reading: `h5py` or `h5netcdf`
- Fusion/correction: `scikit-learn`, `joblib`, `tqdm`, `geopandas`, `regionmask`, `scipy`

---

## Quick start

### 1) Compute SPI and summarize by region
```python
import xarray as xr
from premise.indices import calc_spi
from premise.preprocess import area_mean_by_region
from premise.plotting import plot_violin_groups

ds = xr.open_dataset("pr_daily.nc")     # precipitation (mm/day)
spi3 = calc_spi(ds["pr"], scale=3)

spi3_reg = area_mean_by_region(
    spi3,
    "china_7regions.shp",
    region_field="climate",
)

data_dict = {str(r): spi3_reg.sel(region=r).values for r in spi3_reg.region.values}
fig, ax = plot_violin_groups(data_dict, title="SPI-3 by region", ylabel="SPI-3")
fig.savefig("spi3_violin_regions.png", dpi=300)
```

### 2) Convert raw formats to NetCDF

#### 2.1 Raw binary raster → NetCDF (`premise.binaryio`)
Binary conversion is **config-driven** (minimum required: dtype, dims, coordinates). A typical workflow is:

```python
from premise.binaryio import convert_binary_to_netcdf

convert_binary_to_netcdf(
    bin_path="precip_20000101.bin",
    meta_path="cpc_global_daily.meta",
    out_nc="precip_20000101.nc",
)
```

A minimal `.meta` example (key=value, one per line):
```text
# variable
var=pr
units=mm/day

# data layout
dtype=float32
endian=little
nx=360
ny=180

# coordinates (either explicit arrays or linear spacing)
lon_start=0.0
lon_step=1.0
lat_start=-89.5
lat_step=1.0

# optional time for a single-slice file
time=2000-01-01
calendar=proleptic_gregorian
```

#### 2.2 Daily GeoTIFF → monthly NetCDF (`premise.geotiff`)
```python
from premise.geotiff import convert_daily_geotiff_to_monthly_nc

convert_daily_geotiff_to_monthly_nc(
    in_root="/data/CHIRPSv3/daily/final/sat",
    out_dir="/data/CHIRPSv3/monthly_nc",
    year=2000,
    glob_pattern="chirps-v3.0.sat.*.tif",
    name_regex=r"chirps-v3\.0\.sat\.(\d{4})\.(\d{2})\.(\d{2})\.tif$",
    var_name="pr",
    units="mm/day",
)
```

#### 2.3 GRIB / HDF → NetCDF (via xarray backend)
If `premise.io.to_netcdf()` is enabled in your installation, you can convert in one line once the backend is installed (e.g., `cfgrib` for GRIB2).

```python
import premise as pm

pm.io.to_netcdf("/data/ERA5_tp.grib2", "era5_tp.nc")     # requires cfgrib backend
pm.io.to_netcdf("/data/product.hdf", "product.nc")       # requires h5py/h5netcdf backend
```

---

## Package layout

The repository follows a **src-layout** Python package structure.

| Module | Purpose (high level) |
|---|---|
| `premise.io` | Unified opening/writing of datasets; variable/unit normalization; format conversion wrappers |
| `premise.binaryio` | Raw binary raster → DataArray/NetCDF conversion (config-driven) |
| `premise.geotiff` | Daily GeoTIFF stacks → monthly NetCDF conversion |
| `premise.preprocess` | Masking, regridding, unit conversion, spatial aggregation |
| `premise.indices` | Drought indices (SPI/SPEI/SRI/STI, multi-scale) |
| `premise.extreme_indices` | Extreme climate indices (ETCCDI-style) |
| `premise.metrics` | Continuous + detection-based evaluation metrics |
| `premise.evaluation` | Grid evaluation orchestration and summary tables |
| `premise.plotting` | Publication-oriented plots (maps, distributions, Taylor/performance diagrams) |
| `premise.pointpixel` | Station-to-grid matching and point–pixel evaluation |
| `premise.workflows` | Batch workflows (grid evaluation, indices-by-region, plotting pipelines) |
| `premise.fusion` *(optional)* | Fusion toolbox (RF two-stage, baselines, benchmark suite for ablation) |
| `premise.correction` *(optional)* | Two-stage RF precipitation correction (wet/dry + residual + monthly scaling) |

---

## Scripts (examples)

The `scripts/` folder contains runnable entry scripts you can adapt:

- `convert_binary_to_nc.py` — binary → NetCDF conversion wrapper
- `run_geotiff_to_nc.py` — GeoTIFF → NetCDF wrapper
- `run_grid_evaluation.py` — grid evaluation workflow
- `run_indices_by_region.py` — regional indices workflow
- `plot_performance_diagrams.py` — performance diagram plotting
- `plot_scatter_climatology_premise.py` — climatology scatter diagnostics

---

## Reproducibility

PREMISE is designed to support reproducible experiments through:
- **JSON-driven configuration** for workflows (recommended for publication runs).
- Deterministic random seeds where stochastic sampling is used (e.g., correction/fusion training).
- Standardized variable naming and unit conventions (e.g., precipitation as `pr` in `mm/day`).

---

## Citation

If you use PREMISE in academic work, please cite the associated software paper (Environmental Modelling & Software; in preparation).  
You may also archive a DOI (e.g., via Zenodo) and place it here.

---

## License

Choose a license appropriate for your intended release (e.g., MIT / BSD-3-Clause / Apache-2.0) and add `LICENSE`.

---

## Contributing

Issues and pull requests are welcome. For substantial changes, please open an issue to discuss:
- new dataset format adapters,
- new indices/metrics,
- additional plotting templates,
- workflow extensions.
