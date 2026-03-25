"""Handler registry — maps (category, sub_intent) to handler functions."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from api.handlers.base import HandlerResult
from api.handlers.stats_handler import StatsHandler
from api.handlers.clean_handler import CleanHandler
from api.handlers.transform_handler import TransformHandler
from api.handlers.viz_handler import VizHandler
from api.handlers.feature_handler import FeatureHandler

HandlerFunc = Callable[[pd.DataFrame, dict], HandlerResult]

HANDLER_REGISTRY: dict[tuple[str, str], HandlerFunc] = {
    # Stats
    ("stats", "describe"):          StatsHandler.handle_describe,
    ("stats", "shape"):             StatsHandler.handle_shape,
    ("stats", "null_report"):       StatsHandler.handle_null_report,
    ("stats", "value_counts"):      StatsHandler.handle_value_counts,
    ("stats", "unique_values"):     StatsHandler.handle_unique_values,
    ("stats", "dtypes"):            StatsHandler.handle_dtypes,
    ("stats", "correlation"):       StatsHandler.handle_correlation,
    ("stats", "skewness"):          StatsHandler.handle_skewness,
    ("stats", "outlier_report"):    StatsHandler.handle_outlier_report,
    ("stats", "duplicate_report"):  StatsHandler.handle_duplicate_report,
    ("stats", "cross_tab"):         StatsHandler.handle_cross_tab,
    ("stats", "percentile"):        StatsHandler.handle_percentile,
    ("stats", "normality_test"):    StatsHandler.handle_normality_test,
    ("stats", "class_balance"):     StatsHandler.handle_class_balance,
    ("stats", "top_correlations"):  StatsHandler.handle_top_correlations,
    ("stats", "kurtosis"):          StatsHandler.handle_kurtosis,
    ("stats", "zero_report"):       StatsHandler.handle_zero_report,
    ("stats", "cardinality_report"):StatsHandler.handle_cardinality_report,
    # Clean
    ("clean", "drop_nulls"):        CleanHandler.handle_drop_nulls,
    ("clean", "fill_nulls"):        CleanHandler.handle_fill_nulls,
    ("clean", "remove_duplicates"): CleanHandler.handle_remove_duplicates,
    ("clean", "fix_dtypes"):        CleanHandler.handle_fix_dtypes,
    ("clean", "rename_column"):     CleanHandler.handle_rename_column,
    ("clean", "drop_column"):       CleanHandler.handle_drop_column,
    ("clean", "strip_whitespace"):    CleanHandler.handle_strip_whitespace,
    ("clean", "replace_values"):     CleanHandler.handle_replace_values,
    ("clean", "lowercase_columns"):  CleanHandler.handle_lowercase_columns,
    ("clean", "drop_constant"):      CleanHandler.handle_drop_constant,
    ("clean", "clip_outliers"):       CleanHandler.handle_clip_outliers,
    ("clean", "change_dtype"):       CleanHandler.handle_change_dtype,
    ("clean", "fill_interpolate"):   CleanHandler.handle_fill_interpolate,
    ("clean", "remove_outliers"):    CleanHandler.handle_remove_outliers,
    ("clean", "lowercase_values"):   CleanHandler.handle_lowercase_values,
    ("clean", "map_values"):         CleanHandler.handle_map_values,
    ("clean", "reset_index"):        CleanHandler.handle_reset_index,
    ("clean", "fill_with_value"):    CleanHandler.handle_fill_with_value,
    ("clean", "deduplicate_by"):     CleanHandler.handle_deduplicate_by,
    ("clean", "drop_id_columns"):    CleanHandler.handle_drop_id_columns,
    # Transform
    ("transform", "filter"):         TransformHandler.handle_filter,
    ("transform", "assign_value"):   TransformHandler.handle_assign_value,
    ("transform", "sort"):           TransformHandler.handle_sort,
    ("transform", "groupby_agg"):    TransformHandler.handle_groupby_agg,
    ("transform", "add_column"):     TransformHandler.handle_add_column,
    ("transform", "encode_label"):   TransformHandler.handle_encode_label,
    ("transform", "encode_onehot"):  TransformHandler.handle_encode_onehot,
    ("transform", "scale_minmax"):   TransformHandler.handle_scale_minmax,
    ("transform", "scale_standard"): TransformHandler.handle_scale_standard,
    ("transform", "bin_column"):     TransformHandler.handle_bin_column,
    ("transform", "inject_null"):    TransformHandler.handle_inject_null,
    ("transform", "sample_rows"):    TransformHandler.handle_sample_rows,
    ("transform", "head"):             TransformHandler.handle_head,
    ("transform", "tail"):             TransformHandler.handle_tail,
    ("transform", "pivot"):            TransformHandler.handle_pivot,
    ("transform", "melt"):             TransformHandler.handle_melt,
    ("transform", "scale_robust"):     TransformHandler.handle_scale_robust,
    ("transform", "nlargest"):         TransformHandler.handle_nlargest,
    ("transform", "nsmallest"):        TransformHandler.handle_nsmallest,
    ("transform", "rank"):             TransformHandler.handle_rank,
    ("transform", "cumulative"):       TransformHandler.handle_cumulative,
    ("transform", "rolling"):          TransformHandler.handle_rolling,
    ("transform", "round_values"):     TransformHandler.handle_round_values,
    ("transform", "split_column"):     TransformHandler.handle_split_column,
    ("transform", "concat_columns"):   TransformHandler.handle_concat_columns,
    ("transform", "qcut"):             TransformHandler.handle_qcut,
    # Viz
    ("viz", "bar_chart"):          VizHandler.handle_bar_chart,
    ("viz", "histogram"):          VizHandler.handle_histogram,
    ("viz", "scatter"):            VizHandler.handle_scatter,
    ("viz", "line_chart"):         VizHandler.handle_line_chart,
    ("viz", "box_plot"):           VizHandler.handle_box_plot,
    ("viz", "violin_plot"):        VizHandler.handle_violin_plot,
    ("viz", "heatmap"):            VizHandler.handle_heatmap,
    ("viz", "pie_chart"):          VizHandler.handle_pie_chart,
    ("viz", "pairplot"):           VizHandler.handle_pairplot,
    ("viz", "missing_heatmap"):    VizHandler.handle_missing_heatmap,
    ("viz", "count_plot"):         VizHandler.handle_count_plot,
    ("viz", "time_series"):        VizHandler.handle_time_series,
    ("viz", "bubble_chart"):       VizHandler.handle_bubble_chart,
    ("viz", "treemap"):            VizHandler.handle_treemap,
    ("viz", "sunburst"):           VizHandler.handle_sunburst,
    ("viz", "parallel_coords"):    VizHandler.handle_parallel_coords,
    ("viz", "distribution"):       VizHandler.handle_distribution,
    ("viz", "stacked_bar"):        VizHandler.handle_stacked_bar,
    ("viz", "area_chart"):         VizHandler.handle_area_chart,
    ("viz", "qq_plot"):            VizHandler.handle_qq_plot,
    ("viz", "density_plot"):       VizHandler.handle_density_plot,
    ("viz", "strip_plot"):         VizHandler.handle_strip_plot,
    # Feature
    ("feature", "feature_importance"):  FeatureHandler.handle_feature_importance,
    ("feature", "pca"):                 FeatureHandler.handle_pca,
    ("feature", "correlation_filter"):  FeatureHandler.handle_correlation_filter,
    ("feature", "log_transform"):       FeatureHandler.handle_log_transform,
    ("feature", "variance_filter"):     FeatureHandler.handle_variance_filter,
    ("feature", "polynomial_features"): FeatureHandler.handle_polynomial_features,
    ("feature", "datetime_features"):   FeatureHandler.handle_datetime_features,
    ("feature", "target_encode"):       FeatureHandler.handle_target_encode,
    ("feature", "select_k_best"):       FeatureHandler.handle_select_k_best,
    ("feature", "power_transform"):     FeatureHandler.handle_power_transform,
    ("feature", "ratio_features"):      FeatureHandler.handle_ratio_features,
    ("feature", "frequency_encode"):    FeatureHandler.handle_frequency_encode,
    ("feature", "cyclical_encode"):     FeatureHandler.handle_cyclical_encode,
    ("feature", "sqrt_transform"):      FeatureHandler.handle_sqrt_transform,
    ("feature", "mutual_info"):         FeatureHandler.handle_mutual_info,
    ("feature", "lag_features"):        FeatureHandler.handle_lag_features,
    ("feature", "text_features"):       FeatureHandler.handle_text_features,
    ("feature", "quantile_transform"):  FeatureHandler.handle_quantile_transform,
    ("feature", "diff_features"):       FeatureHandler.handle_diff_features,
    ("feature", "aggregation_features"):FeatureHandler.handle_aggregation_features,
}


def get_handler(category: str, sub_intent: str) -> HandlerFunc | None:
    """Look up handler by (category, sub_intent). Returns None if not found."""
    return HANDLER_REGISTRY.get((category, sub_intent))
