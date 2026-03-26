"""handle_fill_with_value handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fill_with_value(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Fill nulls with a specific constant value (e.g. -1, 0, 'Unknown', 'N/A')."""
    col = params.get("column")
    value = params.get("value", 0)
    result = df.copy()
    before_nulls = int(result.isna().sum().sum())

    if col and col in result.columns:
        result[col] = result[col].fillna(value)
        summary_target = f"in '{col}'"
    else:
        result = result.fillna(value)
        summary_target = "in all columns"

    after_nulls = int(result.isna().sum().sum())
    filled = before_nulls - after_nulls
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Filled {filled:,} nulls with {repr(value)} {summary_target}",
    )
