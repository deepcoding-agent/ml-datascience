"""stats handlers — individual modules."""
from api.handlers.stats.describe import handle_describe
from api.handlers.stats.shape import handle_shape
from api.handlers.stats.null_report import handle_null_report
from api.handlers.stats.value_counts import handle_value_counts
from api.handlers.stats.unique_values import handle_unique_values
from api.handlers.stats.dtypes import handle_dtypes
from api.handlers.stats.correlation import handle_correlation
from api.handlers.stats.skewness import handle_skewness
from api.handlers.stats.outlier_report import handle_outlier_report
from api.handlers.stats.duplicate_report import handle_duplicate_report
from api.handlers.stats.cross_tab import handle_cross_tab
from api.handlers.stats.percentile import handle_percentile
from api.handlers.stats.normality_test import handle_normality_test
from api.handlers.stats.class_balance import handle_class_balance
from api.handlers.stats.top_correlations import handle_top_correlations
from api.handlers.stats.kurtosis import handle_kurtosis
from api.handlers.stats.zero_report import handle_zero_report
from api.handlers.stats.cardinality_report import handle_cardinality_report
from api.handlers.stats.mode_report import handle_mode_report
from api.handlers.stats.variance_report import handle_variance_report
from api.handlers.stats.range_report import handle_range_report
from api.handlers.stats.iqr_report import handle_iqr_report
from api.handlers.stats.z_score_report import handle_z_score_report
from api.handlers.stats.chi2_test import handle_chi2_test
from api.handlers.stats.t_test import handle_t_test
from api.handlers.stats.anova_test import handle_anova_test
from api.handlers.stats.mann_whitney import handle_mann_whitney
from api.handlers.stats.ks_test import handle_ks_test
from api.handlers.stats.frequency_table import handle_frequency_table
from api.handlers.stats.coefficient_variation import handle_coefficient_variation
from api.handlers.stats.correlation_rank import handle_correlation_rank
from api.handlers.stats.entropy_report import handle_entropy_report
from api.handlers.stats.gini_report import handle_gini_report
from api.handlers.stats.missing_pattern import handle_missing_pattern
from api.handlers.stats.quantile_detail import handle_quantile_detail
from api.handlers.stats.group_stats import handle_group_stats
from api.handlers.stats.column_compare import handle_column_compare
from api.handlers.stats.mutual_info_report import handle_mutual_info_report
from api.handlers.stats.summary_extended import handle_summary_extended
from api.handlers.stats.ratio_report import handle_ratio_report
from api.handlers.stats.distribution_fit import handle_distribution_fit
from api.handlers.stats.stability_report import handle_stability_report
from api.handlers.stats.pairwise_stats import handle_pairwise_stats
from api.handlers.stats.time_stats import handle_time_stats
from api.handlers.stats.top_bottom_values import handle_top_bottom_values
from api.handlers.stats.memory_report import handle_memory_report
from api.handlers.stats.cluster_tendency import handle_cluster_tendency
from api.handlers.stats.sparsity_report import handle_sparsity_report
from api.handlers.stats.data_sample import handle_data_sample
from api.handlers.stats.normality_comprehensive import handle_normality_comprehensive

__all__ = [
    "handle_describe",
    "handle_shape",
    "handle_null_report",
    "handle_value_counts",
    "handle_unique_values",
    "handle_dtypes",
    "handle_correlation",
    "handle_skewness",
    "handle_outlier_report",
    "handle_duplicate_report",
    "handle_cross_tab",
    "handle_percentile",
    "handle_normality_test",
    "handle_class_balance",
    "handle_top_correlations",
    "handle_kurtosis",
    "handle_zero_report",
    "handle_cardinality_report",
    "handle_mode_report",
    "handle_variance_report",
    "handle_range_report",
    "handle_iqr_report",
    "handle_z_score_report",
    "handle_chi2_test",
    "handle_t_test",
    "handle_anova_test",
    "handle_mann_whitney",
    "handle_ks_test",
    "handle_frequency_table",
    "handle_coefficient_variation",
    "handle_correlation_rank",
    "handle_entropy_report",
    "handle_gini_report",
    "handle_missing_pattern",
    "handle_quantile_detail",
    "handle_group_stats",
    "handle_column_compare",
    "handle_mutual_info_report",
    "handle_summary_extended",
    "handle_ratio_report",
    "handle_distribution_fit",
    "handle_stability_report",
    "handle_pairwise_stats",
    "handle_time_stats",
    "handle_top_bottom_values",
    "handle_memory_report",
    "handle_cluster_tendency",
    "handle_sparsity_report",
    "handle_data_sample",
    "handle_normality_comprehensive",
]
