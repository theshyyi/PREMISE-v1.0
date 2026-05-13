# Installation

PREMISE v2.1 is distributed as an installable Python package using a `src/` layout and `pyproject.toml` metadata.

## Basic reviewer installation

```bash
git clone https://github.com/theshyyi/PREMISE-v2.1.git
cd PREMISE-v2.1
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Optional dependency groups

```bash
pip install -e ".[geo]"
pip install -e ".[tiff]"
pip install -e ".[grib]"
pip install -e ".[hdf]"
pip install -e ".[fusion]"
pip install -e ".[full]"
```

The core package can be imported without geospatial or machine-learning dependencies. Optional dependencies are only needed for specific workflows such as shapefile masking, GeoTIFF/GRIB/HDF conversion, or fusion experiments.

## Verification

```bash
python -c "import premise; print(premise.__version__)"
python examples/quickstart/basic_usage.py
pytest -q
```
