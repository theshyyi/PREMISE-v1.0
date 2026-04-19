#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot a climatological scatter-density comparison between two products."""

import argparse
import xarray as xr
from matplotlib import rcParams

from premise.plotting import scatter_density_product

rcParams.update({
    "font.family": ["Times New Roman"],
    "font.size": 14,
    "mathtext.fontset": "stix",
})


def parse_args():
    ap = argparse.ArgumentParser(description="Plot a scatter-density climatology figure.")
    ap.add_argument("--ref-file", required=True)
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--var-name", default="pr")
    ap.add_argument("--time-start", default="2000-01-01")
    ap.add_argument("--time-end", default="2020-12-31")
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--title", default="Climatological precipitation comparison")
    ap.add_argument("--x-label", default="Reference climatology")
    ap.add_argument("--y-label", default="Candidate climatology")
    return ap.parse_args()


def main():
    args = parse_args()
    ds_ref = xr.open_dataset(args.ref_file)[args.var_name].sel(time=slice(args.time_start, args.time_end))
    ds_test = xr.open_dataset(args.test_file)[args.var_name].sel(time=slice(args.time_start, args.time_end))
    ref_clim = ds_ref.mean(dim="time", skipna=True)
    test_clim = ds_test.mean(dim="time", skipna=True)
    ref_clim_aligned, test_clim_aligned = xr.align(ref_clim, test_clim, join="inner")
    _, _, stats = scatter_density_product(
        ref_clim_aligned,
        test_clim_aligned,
        x_label=args.x_label,
        y_label=args.y_label,
        title=args.title,
        out_path=args.out_png,
        cmap="gist_rainbow",
        point_size=4.0,
        tick_step=5.0,
    )
    print("Scatter-density statistics:", stats)


if __name__ == "__main__":
    main()
