"""clean handlers — individual modules."""
from api.handlers.clean.drop_nulls import handle_drop_nulls
from api.handlers.clean.fill_nulls import handle_fill_nulls
from api.handlers.clean.remove_duplicates import handle_remove_duplicates
from api.handlers.clean.fix_dtypes import handle_fix_dtypes
from api.handlers.clean.rename_column import handle_rename_column
from api.handlers.clean.drop_column import handle_drop_column
from api.handlers.clean.strip_whitespace import handle_strip_whitespace
from api.handlers.clean.replace_values import handle_replace_values
from api.handlers.clean.lowercase_columns import handle_lowercase_columns
from api.handlers.clean.drop_constant import handle_drop_constant
from api.handlers.clean.clip_outliers import handle_clip_outliers
from api.handlers.clean.change_dtype import handle_change_dtype
from api.handlers.clean.fill_interpolate import handle_fill_interpolate
from api.handlers.clean.remove_outliers import handle_remove_outliers
from api.handlers.clean.lowercase_values import handle_lowercase_values
from api.handlers.clean.map_values import handle_map_values
from api.handlers.clean.reset_index import handle_reset_index
from api.handlers.clean.fill_with_value import handle_fill_with_value
from api.handlers.clean.deduplicate_by import handle_deduplicate_by
from api.handlers.clean.drop_id_columns import handle_drop_id_columns
from api.handlers.clean.fix_numeric_strings import handle_fix_numeric_strings
from api.handlers.clean.clean_column_names import handle_clean_column_names
from api.handlers.clean.remove_empty_rows import handle_remove_empty_rows
from api.handlers.clean.fill_mode import handle_fill_mode
from api.handlers.clean.fill_forward_backward import handle_fill_forward_backward
from api.handlers.clean.fix_boolean import handle_fix_boolean
from api.handlers.clean.fix_encoding import handle_fix_encoding
from api.handlers.clean.remove_html_tags import handle_remove_html_tags
from api.handlers.clean.clean_currency import handle_clean_currency
from api.handlers.clean.standardize_dates import handle_standardize_dates
from api.handlers.clean.remove_non_ascii import handle_remove_non_ascii
from api.handlers.clean.remove_special_chars import handle_remove_special_chars
from api.handlers.clean.normalize_text_case import handle_normalize_text_case
from api.handlers.clean.cap_outliers_percentile import handle_cap_outliers_percentile
from api.handlers.clean.fill_median_by_group import handle_fill_median_by_group
from api.handlers.clean.remove_zero_rows import handle_remove_zero_rows
from api.handlers.clean.remove_negative import handle_remove_negative
from api.handlers.clean.standardize_categories import handle_standardize_categories
from api.handlers.clean.remove_high_null_cols import handle_remove_high_null_cols
from api.handlers.clean.clean_phone_numbers import handle_clean_phone_numbers
from api.handlers.clean.split_name import handle_split_name
from api.handlers.clean.fix_whitespace_names import handle_fix_whitespace_names
from api.handlers.clean.remove_urls import handle_remove_urls
from api.handlers.clean.remove_emails import handle_remove_emails
from api.handlers.clean.fix_mixed_types import handle_fix_mixed_types
from api.handlers.clean.fill_with_distribution import handle_fill_with_distribution
from api.handlers.clean.remove_rare_categories import handle_remove_rare_categories
from api.handlers.clean.dedup_keep_latest import handle_dedup_keep_latest
from api.handlers.clean.fix_date_outliers import handle_fix_date_outliers
from api.handlers.clean.clean_text_whitespace import handle_clean_text_whitespace
from api.handlers.clean.coalesce_columns import handle_coalesce_columns
from api.handlers.clean.anonymize_column import handle_anonymize_column
from api.handlers.clean.parse_json_column import handle_parse_json_column
from api.handlers.clean.trim_text_length import handle_trim_text_length
from api.handlers.clean.round_to_nearest import handle_round_to_nearest

__all__ = [
    "handle_drop_nulls",
    "handle_fill_nulls",
    "handle_remove_duplicates",
    "handle_fix_dtypes",
    "handle_rename_column",
    "handle_drop_column",
    "handle_strip_whitespace",
    "handle_replace_values",
    "handle_lowercase_columns",
    "handle_drop_constant",
    "handle_clip_outliers",
    "handle_change_dtype",
    "handle_fill_interpolate",
    "handle_remove_outliers",
    "handle_lowercase_values",
    "handle_map_values",
    "handle_reset_index",
    "handle_fill_with_value",
    "handle_deduplicate_by",
    "handle_drop_id_columns",
    "handle_fix_numeric_strings",
    "handle_clean_column_names",
    "handle_remove_empty_rows",
    "handle_fill_mode",
    "handle_fill_forward_backward",
    "handle_fix_boolean",
    "handle_fix_encoding",
    "handle_remove_html_tags",
    "handle_clean_currency",
    "handle_standardize_dates",
    "handle_remove_non_ascii",
    "handle_remove_special_chars",
    "handle_normalize_text_case",
    "handle_cap_outliers_percentile",
    "handle_fill_median_by_group",
    "handle_remove_zero_rows",
    "handle_remove_negative",
    "handle_standardize_categories",
    "handle_remove_high_null_cols",
    "handle_clean_phone_numbers",
    "handle_split_name",
    "handle_fix_whitespace_names",
    "handle_remove_urls",
    "handle_remove_emails",
    "handle_fix_mixed_types",
    "handle_fill_with_distribution",
    "handle_remove_rare_categories",
    "handle_dedup_keep_latest",
    "handle_fix_date_outliers",
    "handle_clean_text_whitespace",
    "handle_coalesce_columns",
    "handle_anonymize_column",
    "handle_parse_json_column",
    "handle_trim_text_length",
    "handle_round_to_nearest",
]
