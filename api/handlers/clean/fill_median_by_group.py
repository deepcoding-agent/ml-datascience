"""handle_fill_median_by_group handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fill_median_by_group(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Fill nulls with group-level median."""
    group_col = params.get("group_column") or params.get("column")
    value_col = params.get("value_column")
    if not group_col or group_col not in df.columns:
        return HandlerResult(success=False, error=f"Group column '{group_col}' not found")
    result = df.copy()
    before_nulls = int(result.isna().sum().sum())

    if value_col and value_col in result.columns:
        fill_cols = [value_col]
    else:
        fill_cols = result.select_dtypes(include="number").columns.tolist()

    for c in fill_cols:
        group_median = result.groupby(group_col)[c].transform("median")
        result[c] = result[c].fillna(group_median)

    after_nulls = int(result.isna().sum().sum())
    filled = before_nulls - after_nulls
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Filled {filled:,} nulls with group median (grouped by '{group_col}')",
    )
