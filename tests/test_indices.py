import numpy as np
import pandas as pd
import xarray as xr

from premise.indices import calc_spi


def test_calc_spi_shape_and_contains_dims():
    rng = np.random.RandomState(0)
    time = pd.date_range("2001-01-01", periods=24, freq="MS")
    da = xr.DataArray(
        rng.gamma(shape=2.0, scale=2.0, size=(24, 2, 2)),
        coords={"time": time, "lat": [0, 1], "lon": [10, 11]},
        dims=("time", "lat", "lon"),
        name="pr",
    )
    spi = calc_spi(da, scale=3, is_monthly_input=True)
    assert set(spi.dims) == set(da.dims)
    assert spi.sizes["time"] == da.sizes["time"]
    assert spi.sizes["lat"] == da.sizes["lat"]
    assert spi.sizes["lon"] == da.sizes["lon"]
