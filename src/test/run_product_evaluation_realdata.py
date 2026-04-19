
from __future__ import annotations

from pathlib import Path

from premise.product_evaluation import run_tasks

BASE_DIR = Path(r"I:\PREMISE-v1.0\product_evaluation_output")
REPORT_DIR = BASE_DIR / "_reports"

TASKS = [
    # --------------------------------------------------------
    # 1) Grid -> Grid
    # 参考数据和待评价产品都是 NetCDF
    # 空间分布指标输出 NC；整体/分组指标输出 CSV
    # --------------------------------------------------------
    {
        "name": "grid_to_grid_demo",
        "mode": "grid_to_grid",
        "enabled": False,
        "obs_path": r"I:\PREMISE-v1.0\conversion_output\reference_obs.nc",
        "sim_path": r"I:\PREMISE-v1.0\conversion_output\product_sim.nc",
        "obs_var": "pr",
        "sim_var": "pr",
        "time_range": ("2020-01-01", "2020-12-31"),
        "time_scale": "monthly",     # native / daily / monthly / yearly
        "time_agg": "sum",           # sum / mean / max / min
        "group_by": "month",         # none / month / season / year
        "threshold": 1.0,
        "overall_metrics": ["BIAS", "MAE", "RMSE", "CORR", "KGE", "NSE", "POD", "FAR", "CSI", "FBIAS", "HSS"],
        "spatial_metrics": ["BIAS", "MAE", "RMSE", "CORR", "KGE", "NSE"],
        "out_table_csv": str(BASE_DIR / "grid_to_grid_demo_overall.csv"),
        "out_group_csv": str(BASE_DIR / "grid_to_grid_demo_grouped.csv"),
        "out_spatial_nc": str(BASE_DIR / "grid_to_grid_demo_spatial.nc"),
    },

    # --------------------------------------------------------
    # 2) Table -> Grid
    # 参考数据是实测表格（CSV/XLSX），产品是 NetCDF
    # 输出站点级指标表和分组指标表
    # --------------------------------------------------------
    {
        "name": "table_to_grid_demo",
        "mode": "table_to_grid",
        "enabled": False,
        "obs_table_path": r"I:\PREMISE-v1.0\tests_output\stations_obs.csv",
        "obs_table_format": "csv",   # csv / xlsx；也可省略自动识别
        "sim_nc_path": r"I:\PREMISE-v1.0\conversion_output\product_sim.nc",
        "sim_var": "pr",
        "time_range": ("2020-01-01", "2020-12-31"),
        "time_scale": "monthly",
        "time_agg": "sum",
        "station_col": "station",
        "time_col": "time",
        "lat_col": "lat",
        "lon_col": "lon",
        "value_col": "obs",
        "extract_method": "nearest",
        "group_by": "season",
        "threshold": 1.0,
        "metrics": ["BIAS", "MAE", "RMSE", "CORR", "KGE", "NSE", "POD", "FAR", "CSI", "FBIAS", "HSS"],
        "out_station_csv": str(BASE_DIR / "table_to_grid_demo_station.csv"),
        "out_group_csv": str(BASE_DIR / "table_to_grid_demo_grouped.csv"),
    },
]


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = run_tasks(TASKS, report_dir=REPORT_DIR)

    print("=" * 90)
    for rec in records:
        print(f"[{rec['status']}] {rec['name']} | mode={rec['mode']} | elapsed={rec['elapsed_seconds']} s")
        if rec.get('error_message'):
            print(f"  -> {rec['error_message']}")
    print("=" * 90)
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
