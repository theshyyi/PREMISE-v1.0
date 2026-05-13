# Input data requirements

PREMISE is designed for gridded hydroclimatic products, especially precipitation datasets used in hydrology, climate diagnostics, and product intercomparison.

## Recommended NetCDF conventions

- Coordinates should include `time`, `lat`, and `lon`, or names that can be mapped to these conventions.
- Time should be CF-decodable and monotonic.
- Latitude and longitude should be numeric and explicitly stored as coordinates.
- Precipitation should be expressed as daily rates (`mm day-1`) or accumulated totals (`mm`) with units documented in attributes.
- Missing values should be encoded as NaN or with standard `_FillValue` / `missing_value` attributes.

## Typical variables

| Variable type | Example names | Typical use |
|---|---|---|
| Precipitation | `pr`, `precip`, `precipitation` | Product evaluation, SPI, extremes |
| Temperature | `tas`, `tasmax`, `tasmin` | STI, climate diagnostics |
| Potential evapotranspiration | `pet`, `eto` | SPEI |
| Runoff | `runoff`, `mrro`, `sro` | SRI and hydrological evaluation |
| Soil moisture | `sm`, `mrsos`, `swvl1` | Hydroclimatic diagnostics |

## Spatial and temporal alignment

For product evaluation, reference and candidate datasets should be harmonized to a common grid, time period, temporal resolution, and variable convention before metrics are computed. PREMISE provides utility modules to support this process, but the required workflow depends on the original data format and application domain.
