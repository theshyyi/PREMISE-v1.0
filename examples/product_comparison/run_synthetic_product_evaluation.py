"""Synthetic product-evaluation example."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import xarray as xr

from premise.product_evaluation import evaluate_grid_pair, evaluate_grid_by_group


def main() -> None:
    rng = np.random.RandomState(21)
    time = pd.date_range("2001-01-01", periods=24, freq="MS")
    lat = [30.0, 31.0]
    lon = [110.0, 111.0]
    obs = xr.DataArray(
        rng.gamma(shape=2.0, scale=2.0, size=(24, 2, 2)),
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="reference_pr",
    )
    sim = obs * 1.08 + rng.normal(0.0, 0.2, size=obs.shape)
    sim.name = "candidate_pr"

    overall = evaluate_grid_pair(obs, sim, threshold=2.0, metrics=["BIAS", "MAE", "RMSE", "CORR", "POD", "FAR", "CSI"])
    monthly = evaluate_grid_by_group(obs, sim, group_by="month", threshold=2.0, metrics=["RMSE", "CORR"])

    print("Overall metrics:")
    print(pd.Series(overall).round(3).to_string())
    print("\nMonthly metric table head:")
    print(monthly.head().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
