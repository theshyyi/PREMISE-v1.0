"""Synthetic basin-oriented preprocessing example.

This example avoids external shapefiles and demonstrates lightweight operations
that are always available in the core package.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import xarray as xr

from premise.basin import clip_time, clip_to_bbox, rename_variables


def main() -> None:
    rng = np.random.RandomState(7)
    time = pd.date_range("2001-01-01", periods=12, freq="MS")
    lat = np.linspace(28.0, 32.0, 5)
    lon = np.linspace(108.0, 112.0, 5)
    ds = xr.Dataset(
        {
            "precipitation": (
                ("time", "lat", "lon"),
                rng.gamma(shape=2.0, scale=3.0, size=(len(time), len(lat), len(lon))),
            )
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )

    ds = rename_variables(ds, {"precipitation": "pr"})
    ds = clip_time(ds, start="2001-03-01", end="2001-10-01")
    ds = clip_to_bbox(ds, min_lon=109.0, max_lon=111.0, min_lat=29.0, max_lat=31.0)
    basin_mean = ds["pr"].mean(dim=("lat", "lon"))

    print("Preprocessed dataset dimensions:", dict(ds.sizes))
    print("Synthetic basin mean length:", basin_mean.sizes["time"])


if __name__ == "__main__":
    main()
