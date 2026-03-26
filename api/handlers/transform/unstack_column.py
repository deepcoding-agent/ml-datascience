"""handle_unstack_column handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_unstack_column(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Unstack/pivot a column to wide format. Requires index, column, and value cols."""
    index_col = params.get("index")
    col_col = params.get("column")
    value_col = params.get("value")

    if not col_col or col_col not in df.columns:
        return HandlerResult(success=False, error="Specify column= for the column to unstack")

    if not index_col:
        candidates = [c for c in df.columns if c != col_col and c != value_col]
        index_col = candidates[0] if candidates else None
    if not value_col:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        value_col = num_cols[0] if num_cols else None

    if index_col is None or value_col is None:
        return HandlerResult(success=False, error="Need index, column, and value params")

    try:
        result = df.pivot_table(index=index_col, columns=col_col,
                                 values=value_col, aggfunc="first").reset_index()
        result.columns = [str(c) for c in result.columns]
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Unstacked '{col_col}' → {len(result.columns)-1} new columns, {len(result)} rows",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Unstack error: {e}")
