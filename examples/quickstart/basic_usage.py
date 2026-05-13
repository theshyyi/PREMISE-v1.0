"""Minimal synthetic quickstart for PREMISE."""

import numpy as np
import pandas as pd
import xarray as xr

from premise.indices import calc_spi
from premise.metrics import rmse


time = pd.date_range("2001-01-01", periods=24, freq="MS")
lat = [30.0, 31.0]
lon = [110.0, 111.0]
values = np.random.RandomState(42).gamma(shape=2.0, scale=2.0, size=(24, 2, 2))

pr = xr.DataArray(values, coords={"time": time, "lat": lat, "lon": lon}, dims=("time", "lat", "lon"), name="pr")
spi3 = calc_spi(pr, scale=3, is_monthly_input=True)

print("SPI-3 computed with shape:", spi3.shape)
print("RMSE(pr, pr) =", rmse(pr, pr))
