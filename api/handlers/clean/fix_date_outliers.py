"""handle_fix_date_outliers handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fix_date_outliers(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Remove or clip dates outside a valid range."""
    col = params.get("column")
    min_date = params.get("min_date", "1900-01-01")
    max_date = params.get("max_date", "2099-12-31")
    action = params.get("action", "remove")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    result = df.copy()
    result[col] = pd.to_datetime(result[col], errors="coerce", format="mixed")
    lo = pd.Timestamp(min_date)
    hi = pd.Timestamp(max_date)
    before = len(result)

    if action == "clip":
        result[col] = result[col].clip(lower=lo, upper=hi)
        summary = f"Clipped dates in '{col}' to [{min_date}, {max_date}]"
    else:  # remove
        mask = (result[col] >= lo) & (result[col] <= hi) | result[col].isna()
        result = result[mask].reset_index(drop=True)
        removed = before - len(result)
        summary = f"Removed {removed:,} rows with dates outside [{min_date}, {max_date}] in '{col}'"

    return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)
