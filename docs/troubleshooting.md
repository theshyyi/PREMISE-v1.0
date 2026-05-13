# Troubleshooting

## `ModuleNotFoundError: premise`

Install the package from the repository root:

```bash
pip install -e ".[dev]"
```

Alternatively, for temporary local testing:

```bash
export PYTHONPATH=src
```

## Optional dependency errors

Some workflows require optional dependencies. Install the relevant group:

```bash
pip install -e ".[geo]"
pip install -e ".[tiff]"
pip install -e ".[grib]"
pip install -e ".[hdf]"
```

## Real data do not align

Check that reference and candidate datasets use the same coordinate names, units, calendar, spatial grid, and time period. Harmonize these before product evaluation.

## Tests unexpectedly collect old local tests

The repository-level pytest configuration restricts collection to the `tests/` directory. Avoid placing local smoke tests under `src/` unless they are intentionally maintained as package tests.
