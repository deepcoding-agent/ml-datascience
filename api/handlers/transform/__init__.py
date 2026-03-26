"""transform handlers — individual modules."""
from api.handlers.transform.filter import handle_filter
from api.handlers.transform.assign_value import handle_assign_value
from api.handlers.transform.sort import handle_sort
from api.handlers.transform.groupby_agg import handle_groupby_agg
from api.handlers.transform.add_column import handle_add_column
from api.handlers.transform.encode_label import handle_encode_label
from api.handlers.transform.encode_onehot import handle_encode_onehot
from api.handlers.transform.scale_minmax import handle_scale_minmax
from api.handlers.transform.scale_standard import handle_scale_standard
from api.handlers.transform.bin_column import handle_bin_column
from api.handlers.transform.inject_null import handle_inject_null
from api.handlers.transform.sample_rows import handle_sample_rows
from api.handlers.transform.head import handle_head
from api.handlers.transform.tail import handle_tail
from api.handlers.transform.pivot import handle_pivot
from api.handlers.transform.melt import handle_melt
from api.handlers.transform.scale_robust import handle_scale_robust
from api.handlers.transform.nlargest import handle_nlargest
from api.handlers.transform.nsmallest import handle_nsmallest
from api.handlers.transform.rank import handle_rank
from api.handlers.transform.cumulative import handle_cumulative
from api.handlers.transform.rolling import handle_rolling
from api.handlers.transform.round_values import handle_round_values
from api.handlers.transform.split_column import handle_split_column
from api.handlers.transform.concat_columns import handle_concat_columns
from api.handlers.transform.qcut import handle_qcut
from api.handlers.transform.merge import handle_merge
from api.handlers.transform.transpose import handle_transpose
from api.handlers.transform.drop_rows import handle_drop_rows
from api.handlers.transform.shuffle import handle_shuffle
from api.handlers.transform.train_test_split import handle_train_test_split
from api.handlers.transform.clip import handle_clip
from api.handlers.transform.where import handle_where
from api.handlers.transform.explode import handle_explode
from api.handlers.transform.encode_binary import handle_encode_binary
from api.handlers.transform.pct_change import handle_pct_change
from api.handlers.transform.normalize_pct import handle_normalize_pct
from api.handlers.transform.apply_expr import handle_apply_expr
from api.handlers.transform.flatten_columns import handle_flatten_columns
from api.handlers.transform.resample import handle_resample
from api.handlers.transform.cross_join import handle_cross_join
from api.handlers.transform.encode_ordinal import handle_encode_ordinal
from api.handlers.transform.shift_column import handle_shift_column
from api.handlers.transform.winsorize import handle_winsorize
from api.handlers.transform.stack_columns import handle_stack_columns
from api.handlers.transform.unstack_column import handle_unstack_column
from api.handlers.transform.reorder_columns import handle_reorder_columns
from api.handlers.transform.duplicate_column import handle_duplicate_column
from api.handlers.transform.fill_forward import handle_fill_forward
from api.handlers.transform.interpolate_values import handle_interpolate_values
from api.handlers.transform.chunk_rows import handle_chunk_rows
from api.handlers.transform.conditional_column import handle_conditional_column
from api.handlers.transform.datetime_parse import handle_datetime_parse
from api.handlers.transform.datetime_format import handle_datetime_format
from api.handlers.transform.cast_columns import handle_cast_columns
from api.handlers.transform.string_slice import handle_string_slice

__all__ = [
    "handle_filter",
    "handle_assign_value",
    "handle_sort",
    "handle_groupby_agg",
    "handle_add_column",
    "handle_encode_label",
    "handle_encode_onehot",
    "handle_scale_minmax",
    "handle_scale_standard",
    "handle_bin_column",
    "handle_inject_null",
    "handle_sample_rows",
    "handle_head",
    "handle_tail",
    "handle_pivot",
    "handle_melt",
    "handle_scale_robust",
    "handle_nlargest",
    "handle_nsmallest",
    "handle_rank",
    "handle_cumulative",
    "handle_rolling",
    "handle_round_values",
    "handle_split_column",
    "handle_concat_columns",
    "handle_qcut",
    "handle_merge",
    "handle_transpose",
    "handle_drop_rows",
    "handle_shuffle",
    "handle_train_test_split",
    "handle_clip",
    "handle_where",
    "handle_explode",
    "handle_encode_binary",
    "handle_pct_change",
    "handle_normalize_pct",
    "handle_apply_expr",
    "handle_flatten_columns",
    "handle_resample",
    "handle_cross_join",
    "handle_encode_ordinal",
    "handle_shift_column",
    "handle_winsorize",
    "handle_stack_columns",
    "handle_unstack_column",
    "handle_reorder_columns",
    "handle_duplicate_column",
    "handle_fill_forward",
    "handle_interpolate_values",
    "handle_chunk_rows",
    "handle_conditional_column",
    "handle_datetime_parse",
    "handle_datetime_format",
    "handle_cast_columns",
    "handle_string_slice",
]
