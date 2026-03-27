"""feature handlers — individual modules."""
from api.handlers.feature.feature_importance import handle_feature_importance
from api.handlers.feature.pca import handle_pca
from api.handlers.feature.correlation_filter import handle_correlation_filter
from api.handlers.feature.log_transform import handle_log_transform
from api.handlers.feature.variance_filter import handle_variance_filter
from api.handlers.feature.polynomial_features import handle_polynomial_features
from api.handlers.feature.datetime_features import handle_datetime_features
from api.handlers.feature.target_encode import handle_target_encode
from api.handlers.feature.select_k_best import handle_select_k_best
from api.handlers.feature.power_transform import handle_power_transform
from api.handlers.feature.ratio_features import handle_ratio_features
from api.handlers.feature.frequency_encode import handle_frequency_encode
from api.handlers.feature.cyclical_encode import handle_cyclical_encode
from api.handlers.feature.sqrt_transform import handle_sqrt_transform
from api.handlers.feature.mutual_info import handle_mutual_info
from api.handlers.feature.lag_features import handle_lag_features
from api.handlers.feature.text_features import handle_text_features
from api.handlers.feature.quantile_transform import handle_quantile_transform
from api.handlers.feature.diff_features import handle_diff_features
from api.handlers.feature.aggregation_features import handle_aggregation_features
from api.handlers.feature.log1p_transform import handle_log1p_transform
from api.handlers.feature.interaction_features import handle_interaction_features
from api.handlers.feature.bin_numeric import handle_bin_numeric
from api.handlers.feature.abs_transform import handle_abs_transform
from api.handlers.feature.reciprocal_transform import handle_reciprocal_transform
from api.handlers.feature.count_encode import handle_count_encode
from api.handlers.feature.rare_category_encode import handle_rare_category_encode
from api.handlers.feature.is_null_features import handle_is_null_features
from api.handlers.feature.is_zero_features import handle_is_zero_features
from api.handlers.feature.rank_transform import handle_rank_transform
from api.handlers.feature.kbins_discretize import handle_kbins_discretize
from api.handlers.feature.hash_encode import handle_hash_encode
from api.handlers.feature.ordinal_encode import handle_ordinal_encode
from api.handlers.feature.label_binarize import handle_label_binarize
from api.handlers.feature.exponential_transform import handle_exponential_transform
from api.handlers.feature.sin_cos_hour import handle_sin_cos_hour
from api.handlers.feature.is_weekend import handle_is_weekend
from api.handlers.feature.is_holiday import handle_is_holiday
from api.handlers.feature.time_since import handle_time_since
from api.handlers.feature.distance_from_mean import handle_distance_from_mean
from api.handlers.feature.clip_features import handle_clip_features
from api.handlers.feature.auto_feature_select import handle_auto_feature_select
from api.handlers.feature.feature_cross import handle_feature_cross
from api.handlers.feature.rolling_stats_features import handle_rolling_stats_features
from api.handlers.feature.ewm_features import handle_ewm_features
from api.handlers.feature.target_binary_encode import handle_target_binary_encode
from api.handlers.feature.zscore_features import handle_zscore_features
from api.handlers.feature.boxcox_transform import handle_boxcox_transform
from api.handlers.feature.yeo_johnson_transform import handle_yeo_johnson_transform
from api.handlers.feature.winsorize import handle_winsorize
from api.handlers.feature.svd_features import handle_svd_features
from api.handlers.feature.percentile_rank import handle_percentile_rank
from api.handlers.feature.string_length_features import handle_string_length_features
from api.handlers.feature.cumulative_features import handle_cumulative_features
from api.handlers.feature.woe_encode import handle_woe_encode
from api.handlers.feature.rfe_select import handle_rfe_select
from api.handlers.feature.mutual_info_select import handle_mutual_info_select
from api.handlers.feature.variance_threshold import handle_variance_threshold
from api.handlers.feature.correlation_select import handle_correlation_select
from api.handlers.feature.boruta_select import handle_boruta_select
from api.handlers.feature.rolling_window import handle_rolling_window
from api.handlers.feature.expanding_window import handle_expanding_window
from api.handlers.feature.seasonal_features import handle_seasonal_features

__all__ = [
    "handle_feature_importance",
    "handle_pca",
    "handle_correlation_filter",
    "handle_log_transform",
    "handle_variance_filter",
    "handle_polynomial_features",
    "handle_datetime_features",
    "handle_target_encode",
    "handle_select_k_best",
    "handle_power_transform",
    "handle_ratio_features",
    "handle_frequency_encode",
    "handle_cyclical_encode",
    "handle_sqrt_transform",
    "handle_mutual_info",
    "handle_lag_features",
    "handle_text_features",
    "handle_quantile_transform",
    "handle_diff_features",
    "handle_aggregation_features",
    "handle_log1p_transform",
    "handle_interaction_features",
    "handle_bin_numeric",
    "handle_abs_transform",
    "handle_reciprocal_transform",
    "handle_count_encode",
    "handle_rare_category_encode",
    "handle_is_null_features",
    "handle_is_zero_features",
    "handle_rank_transform",
    "handle_kbins_discretize",
    "handle_hash_encode",
    "handle_ordinal_encode",
    "handle_label_binarize",
    "handle_exponential_transform",
    "handle_sin_cos_hour",
    "handle_is_weekend",
    "handle_is_holiday",
    "handle_time_since",
    "handle_distance_from_mean",
    "handle_clip_features",
    "handle_auto_feature_select",
    "handle_feature_cross",
    "handle_rolling_stats_features",
    "handle_ewm_features",
    "handle_target_binary_encode",
    "handle_zscore_features",
    "handle_boxcox_transform",
    "handle_yeo_johnson_transform",
    "handle_winsorize",
    "handle_svd_features",
    "handle_percentile_rank",
    "handle_string_length_features",
    "handle_cumulative_features",
    "handle_woe_encode",
    "handle_rfe_select",
    "handle_mutual_info_select",
    "handle_variance_threshold",
    "handle_correlation_select",
    "handle_boruta_select",
    "handle_rolling_window",
    "handle_expanding_window",
    "handle_seasonal_features",
]
