#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate seasonal, monthly, and elevation-based performance diagrams."""

import argparse

from premise.evaluation import (
    compute_pod_far_temporal_from_directory,
    compute_pod_far_by_elevation_from_directory,
)
from premise.plotting import (
    plot_performance_diagram_seasons,
    plot_performance_diagram_months,
    plot_performance_diagram_elevation,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Generate performance diagrams for precipitation products.")
    ap.add_argument("--nc-dir", required=True)
    ap.add_argument("--ref-name", required=True)
    ap.add_argument("--shp", required=True)
    ap.add_argument("--dem-nc", required=True)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--time-start", default="2000-01-01")
    ap.add_argument("--time-end", default="2022-12-31")
    ap.add_argument("--out-prefix", default="performance")
    return ap.parse_args()


def main():
    args = parse_args()
    season_stats, month_stats, products = compute_pod_far_temporal_from_directory(
        args.nc_dir,
        args.ref_name,
        shp_path=args.shp,
        threshold=args.threshold,
        time_range=(args.time_start, args.time_end),
    )
    plot_performance_diagram_seasons(season_stats, products, out_path=f"{args.out_prefix}_seasons.png")
    plot_performance_diagram_months(month_stats, products, out_path=f"{args.out_prefix}_months.png")

    elev_labels, elev_stats, products2 = compute_pod_far_by_elevation_from_directory(
        args.nc_dir,
        args.ref_name,
        dem_nc=args.dem_nc,
        elev_bins=[(0, 200), (0, 500), (500, 1000), (1000, 1500), (1500, 3000), (3000, 10000)],
        threshold=max(args.threshold, 0.1),
        year_start=int(args.time_start[:4]),
        year_end=int(args.time_end[:4]),
        shp_path=args.shp,
    )
    plot_performance_diagram_elevation(elev_labels, elev_stats, products2, out_path=f"{args.out_prefix}_elevation.png")


if __name__ == "__main__":
    main()
