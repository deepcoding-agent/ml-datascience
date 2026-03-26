"""handle_groupby_agg handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_groupby_agg(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    agg = params.get("agg", "count")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    if agg == "count":
        result = df.groupby(col, dropna=False).size().reset_index(name="count")
    else:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns to aggregate")
        result = df.groupby(col, dropna=False)[num_cols].agg(agg).reset_index()
    result = result.sort_values(result.columns[-1], ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, output_type="query",
                         summary=f"Grouped by '{col}' with {agg}")
