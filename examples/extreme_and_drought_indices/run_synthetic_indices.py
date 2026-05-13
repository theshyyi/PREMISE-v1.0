"""Synthetic hydroclimatic and precipitation-extreme index example."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import xarray as xr

from premise.indices import calc_spi
from premise.climate_indices import rx1day, rx5day, prcptot


def main() -> None:
    rng = np.random.RandomState(11)
    daily_time = pd.date_range("2001-01-01", periods=365, freq="D")
    monthly_time = pd.date_range("2001-01-01", periods=36, freq="MS")
    lat = [30.0, 31.0]
    lon = [110.0, 111.0]

    daily_pr = xr.DataArray(
        rng.gamma(shape=1.5, scale=4.0, size=(365, 2, 2)),
        coords={"time": daily_time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="daily_pr",
    )
    monthly_pr = xr.DataArray(
        rng.gamma(shape=2.0, scale=20.0, size=(36, 2, 2)),
        coords={"time": monthly_time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="monthly_pr",
    )

    spi3 = calc_spi(monthly_pr, scale=3, is_monthly_input=True)
    out = xr.Dataset({
        "SPI_3": spi3,
        "Rx1day": rx1day(daily_pr, freq="YE"),
        "Rx5day": rx5day(daily_pr, freq="YE"),
        "PRCPTOT": prcptot(daily_pr, freq="YE"),
    })

    print("Computed variables:", list(out.data_vars))
    print("SPI-3 shape:", out["SPI_3"].shape)


if __name__ == "__main__":
    main()
