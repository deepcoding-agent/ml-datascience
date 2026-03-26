"""viz handlers — individual modules."""
from api.handlers.viz.bar_chart import handle_bar_chart
from api.handlers.viz.histogram import handle_histogram
from api.handlers.viz.scatter import handle_scatter
from api.handlers.viz.line_chart import handle_line_chart
from api.handlers.viz.box_plot import handle_box_plot
from api.handlers.viz.violin_plot import handle_violin_plot
from api.handlers.viz.heatmap import handle_heatmap
from api.handlers.viz.pie_chart import handle_pie_chart
from api.handlers.viz.pairplot import handle_pairplot
from api.handlers.viz.missing_heatmap import handle_missing_heatmap
from api.handlers.viz.count_plot import handle_count_plot
from api.handlers.viz.time_series import handle_time_series
from api.handlers.viz.bubble_chart import handle_bubble_chart
from api.handlers.viz.treemap import handle_treemap
from api.handlers.viz.sunburst import handle_sunburst
from api.handlers.viz.parallel_coords import handle_parallel_coords
from api.handlers.viz.distribution import handle_distribution
from api.handlers.viz.stacked_bar import handle_stacked_bar
from api.handlers.viz.area_chart import handle_area_chart
from api.handlers.viz.qq_plot import handle_qq_plot
from api.handlers.viz.density_plot import handle_density_plot
from api.handlers.viz.strip_plot import handle_strip_plot
from api.handlers.viz.donut_chart import handle_donut_chart
from api.handlers.viz.grouped_bar import handle_grouped_bar
from api.handlers.viz.percent_bar import handle_percent_bar
from api.handlers.viz.pareto_chart import handle_pareto_chart
from api.handlers.viz.waterfall_chart import handle_waterfall_chart
from api.handlers.viz.funnel_chart import handle_funnel_chart
from api.handlers.viz.radar_chart import handle_radar_chart
from api.handlers.viz.lollipop_chart import handle_lollipop_chart
from api.handlers.viz.dual_axis import handle_dual_axis
from api.handlers.viz.histogram_2d import handle_histogram_2d
from api.handlers.viz.ecdf_plot import handle_ecdf_plot
from api.handlers.viz.step_chart import handle_step_chart
from api.handlers.viz.error_bar_chart import handle_error_bar_chart
from api.handlers.viz.ridgeline import handle_ridgeline
from api.handlers.viz.dot_plot import handle_dot_plot
from api.handlers.viz.polar_chart import handle_polar_chart
from api.handlers.viz.sankey_chart import handle_sankey_chart
from api.handlers.viz.candlestick import handle_candlestick
from api.handlers.viz.contour_plot import handle_contour_plot
from api.handlers.viz.cumulative_line import handle_cumulative_line
from api.handlers.viz.null_bar import handle_null_bar
from api.handlers.viz.top_n_bar import handle_top_n_bar
from api.handlers.viz.correlation_scatter import handle_correlation_scatter
from api.handlers.viz.range_plot import handle_range_plot
from api.handlers.viz.swarm_plot import handle_swarm_plot
from api.handlers.viz.comparison_bar import handle_comparison_bar
from api.handlers.viz.marimekko import handle_marimekko
from api.handlers.viz.gauge_chart import handle_gauge_chart

__all__ = [
    "handle_bar_chart",
    "handle_histogram",
    "handle_scatter",
    "handle_line_chart",
    "handle_box_plot",
    "handle_violin_plot",
    "handle_heatmap",
    "handle_pie_chart",
    "handle_pairplot",
    "handle_missing_heatmap",
    "handle_count_plot",
    "handle_time_series",
    "handle_bubble_chart",
    "handle_treemap",
    "handle_sunburst",
    "handle_parallel_coords",
    "handle_distribution",
    "handle_stacked_bar",
    "handle_area_chart",
    "handle_qq_plot",
    "handle_density_plot",
    "handle_strip_plot",
    "handle_donut_chart",
    "handle_grouped_bar",
    "handle_percent_bar",
    "handle_pareto_chart",
    "handle_waterfall_chart",
    "handle_funnel_chart",
    "handle_radar_chart",
    "handle_lollipop_chart",
    "handle_dual_axis",
    "handle_histogram_2d",
    "handle_ecdf_plot",
    "handle_step_chart",
    "handle_error_bar_chart",
    "handle_ridgeline",
    "handle_dot_plot",
    "handle_polar_chart",
    "handle_sankey_chart",
    "handle_candlestick",
    "handle_contour_plot",
    "handle_cumulative_line",
    "handle_null_bar",
    "handle_top_n_bar",
    "handle_correlation_scatter",
    "handle_range_plot",
    "handle_swarm_plot",
    "handle_comparison_bar",
    "handle_marimekko",
    "handle_gauge_chart",
]
