#!/usr/bin/env python
# -*- coding: utf-8 -*-

from premise.evaluation import (
    compute_pod_far_temporal_from_directory,
    compute_pod_far_by_elevation_from_directory,
)
from premise.plotting import (
    plot_performance_diagram_seasons,
    plot_performance_diagram_months,
    plot_performance_diagram_elevation,
)

NC_DIR = "/home/ud202380664/PRE_MERGE/TIMEFIX/Finish"
REF_NAME = "CMFDV2"
SHP = "/home/ud202380664/CHINA/ObeservationData/Chinese_Climate/Chinese_climate.shp"
DEM_NC = "/home/ud202380664/CHINA/ObeservationData/etopo2_new.nc"

def main():
    # 1) 季节 + 月度
    season_stats, month_stats, products = compute_pod_far_temporal_from_directory(
        NC_DIR,
        REF_NAME,
        shp_path=SHP,
        threshold=1.0,
        time_range=("2000-01-01", "2022-12-31"),
    )
    plot_performance_diagram_seasons(
        season_stats,
        products,
        out_path="performance_seasons_1mm.png",
    )
    plot_performance_diagram_months(
        month_stats,
        products,
        out_path="performance_months_1mm.png",
    )

    # 2) 按海拔带
    elev_labels, elev_stats, products2 = compute_pod_far_by_elevation_from_directory(
        NC_DIR,
        REF_NAME,
        dem_nc=DEM_NC,
        elev_bins=[(0, 200), (0, 500), (500, 1000), (1000, 1500), (1500, 3000), (3000, 10000)],
        threshold=0.1,
        year_start=2000,
        year_end=2022,
        shp_path=SHP,
    )
    plot_performance_diagram_elevation(
        elev_labels,
        elev_stats,
        products2,
        out_path="POD_FAR_by_elevation.png",
    )

if __name__ == "__main__":
    main()
