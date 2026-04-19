# Basin forcing preparation workflow

This example slot is reserved for a reproducible basin-oriented preprocessing case.

Recommended steps:

1. Query or register a hydroclimatic source in `premise.acquisition`.
2. Convert raw files to standardized NetCDF through `premise.conversion`.
3. Clip and aggregate the product to a target basin using `premise.basin`.
4. Export forcing datasets for downstream hydrological modelling.
