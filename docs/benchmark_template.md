# Benchmark reporting template

The manuscript and repository should report representative runtime information for a small set of common tasks. A suggested benchmark table is shown below.

| Task | Input size | Hardware | Runtime | Peak memory | Notes |
|---|---|---|---|---|---|
| Binary to NetCDF conversion | 1 daily raster | CPU model / RAM | fill | fill | specify dtype and grid |
| Basin clipping | 1 product, N years | CPU model / RAM | fill | fill | specify polygon count |
| Regridding | source grid to target grid | CPU model / RAM | fill | fill | specify interpolation |
| SPI or SPEI calculation | daily or monthly time series | CPU model / RAM | fill | fill | specify scale |
| Grid evaluation | reference vs candidate products | CPU model / RAM | fill | fill | specify metrics |

A short narrative note should also state the software environment, Python version, operating system, and whether optional dependencies were enabled.
