"""handle_group_stats handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_group_stats(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Descriptive stats per group (groupby + describe)."""
    col = params.get("column")
    value_col = params.get("value_column")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not col or col not in df.columns:
        col = cat_cols[0] if cat_cols else None
    if not value_col or value_col not in df.columns:
        value_col = num_cols[0] if num_cols else None
    if not col or not value_col:
        return HandlerResult(success=False, error="Need a categorical group column and a numeric value column")

    grouped = df.groupby(col)[value_col].agg(["count", "mean", "std", "min", "median", "max"]).round(4)
    result = grouped.reset_index()
    return HandlerResult(success=True, result_df=result,
                         summary=f"Stats of '{value_col}' grouped by '{col}' ({len(result)} groups)")
