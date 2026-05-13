from .styles import setup_mpl_fonts
from .maps import plot_spatial_field, plot_multi_spatial_fields
from .timeseries import plot_timeseries_lines
from .distributions import plot_metric_heatmap, plot_grouped_bar, plot_boxplot_groups, plot_violin_groups
from .diagnostics import (
    scatter_density_product,
    multi_scatter_density_products,
    time_group_scatter_density_product,
    scatter_density_product_by_regions,
    plot_taylor_diagram,
    plot_sal_scatter,
    draw_performance_background,
    plot_performance_diagram_seasons,
    plot_performance_diagram_months,
    plot_performance_diagram_elevation,
    plot_performance_diagram_regions,
    save_performance_diagrams_regions_by_group,
)
from .ranking import plot_ranking_bar, plot_ranking_score_heatmap
from .auto import suggest_visualization
from .pipeline import run_visualization_task, run_visualization_tasks
