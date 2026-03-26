"""analysis handlers — individual modules."""
from api.handlers.analysis.compare_extremes import handle_compare_extremes
from api.handlers.analysis.deep_profile import handle_deep_profile
from api.handlers.analysis.group_insights import handle_group_insights
from api.handlers.analysis.anomaly_detect import handle_anomaly_detect
from api.handlers.analysis.data_quality import handle_data_quality
from api.handlers.analysis.correlation_insights import handle_correlation_insights
from api.handlers.analysis.compare_columns import handle_compare_columns
from api.handlers.analysis.trend_detect import handle_trend_detect
from api.handlers.analysis.segment_analysis import handle_segment_analysis
from api.handlers.analysis.auto_eda import handle_auto_eda
from api.handlers.analysis.cluster_kmeans import handle_cluster_kmeans
from api.handlers.analysis.pca_2d import handle_pca_2d
from api.handlers.analysis.outlier_isolation_forest import handle_outlier_isolation_forest
from api.handlers.analysis.feature_selection_auto import handle_feature_selection_auto
from api.handlers.analysis.distribution_analysis import handle_distribution_analysis
from api.handlers.analysis.missing_value_analysis import handle_missing_value_analysis
from api.handlers.analysis.categorical_analysis import handle_categorical_analysis
from api.handlers.analysis.numeric_summary import handle_numeric_summary
from api.handlers.analysis.hypothesis_test import handle_hypothesis_test
from api.handlers.analysis.regression_quick import handle_regression_quick
from api.handlers.analysis.top_n_analysis import handle_top_n_analysis
from api.handlers.analysis.bottom_n_analysis import handle_bottom_n_analysis
from api.handlers.analysis.percentile_analysis import handle_percentile_analysis
from api.handlers.analysis.variance_analysis import handle_variance_analysis
from api.handlers.analysis.change_point_detect import handle_change_point_detect
from api.handlers.analysis.target_analysis import handle_target_analysis
from api.handlers.analysis.multicollinearity_check import handle_multicollinearity_check
from api.handlers.analysis.ab_test import handle_ab_test
from api.handlers.analysis.pareto_analysis import handle_pareto_analysis
from api.handlers.analysis.data_completeness import handle_data_completeness
from api.handlers.analysis.seasonality_detect import handle_seasonality_detect
from api.handlers.analysis.cluster_profile import handle_cluster_profile
from api.handlers.analysis.feature_interaction import handle_feature_interaction
from api.handlers.analysis.cohort_analysis import handle_cohort_analysis
from api.handlers.analysis.rfm_analysis import handle_rfm_analysis
from api.handlers.analysis.gap_analysis import handle_gap_analysis
from api.handlers.analysis.benchmark_compare import handle_benchmark_compare
from api.handlers.analysis.sensitivity_analysis import handle_sensitivity_analysis
from api.handlers.analysis.correlation_network import handle_correlation_network
from api.handlers.analysis.feature_drift import handle_feature_drift
from api.handlers.analysis.sample_bias_check import handle_sample_bias_check
from api.handlers.analysis.effect_size import handle_effect_size
from api.handlers.analysis.bootstrap_ci import handle_bootstrap_ci
from api.handlers.analysis.cross_correlation import handle_cross_correlation
from api.handlers.analysis.survival_curve import handle_survival_curve
from api.handlers.analysis.concentration_analysis import handle_concentration_analysis
from api.handlers.analysis.diminishing_returns import handle_diminishing_returns
from api.handlers.analysis.categorical_target_crosstab import handle_categorical_target_crosstab
from api.handlers.analysis.prediction_baseline import handle_prediction_baseline
from api.handlers.analysis.data_readiness_score import handle_data_readiness_score

__all__ = [
    "handle_compare_extremes",
    "handle_deep_profile",
    "handle_group_insights",
    "handle_anomaly_detect",
    "handle_data_quality",
    "handle_correlation_insights",
    "handle_compare_columns",
    "handle_trend_detect",
    "handle_segment_analysis",
    "handle_auto_eda",
    "handle_cluster_kmeans",
    "handle_pca_2d",
    "handle_outlier_isolation_forest",
    "handle_feature_selection_auto",
    "handle_distribution_analysis",
    "handle_missing_value_analysis",
    "handle_categorical_analysis",
    "handle_numeric_summary",
    "handle_hypothesis_test",
    "handle_regression_quick",
    "handle_top_n_analysis",
    "handle_bottom_n_analysis",
    "handle_percentile_analysis",
    "handle_variance_analysis",
    "handle_change_point_detect",
    "handle_target_analysis",
    "handle_multicollinearity_check",
    "handle_ab_test",
    "handle_pareto_analysis",
    "handle_data_completeness",
    "handle_seasonality_detect",
    "handle_cluster_profile",
    "handle_feature_interaction",
    "handle_cohort_analysis",
    "handle_rfm_analysis",
    "handle_gap_analysis",
    "handle_benchmark_compare",
    "handle_sensitivity_analysis",
    "handle_correlation_network",
    "handle_feature_drift",
    "handle_sample_bias_check",
    "handle_effect_size",
    "handle_bootstrap_ci",
    "handle_cross_correlation",
    "handle_survival_curve",
    "handle_concentration_analysis",
    "handle_diminishing_returns",
    "handle_categorical_target_crosstab",
    "handle_prediction_baseline",
    "handle_data_readiness_score",
]
