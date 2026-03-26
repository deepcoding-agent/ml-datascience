"""handle_fill_interpolate handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fill_interpolate(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Fill nulls via interpolation — linear, ffill (forward), or bfill (backward)."""
    method = params.get("method", "linear")
    col = params.get("column")
    result = df.copy()
    before_nulls = int(result.isna().sum().sum())

    if method == "ffill":
        if col and col in result.columns:
            result[col] = result[col].ffill()
        else:
            result = result.ffill()
    elif method == "bfill":
        if col and col in result.columns:
            result[col] = result[col].bfill()
        else:
            result = result.bfill()
    else:  # linear
        num_cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
        for c in num_cols:
            result[c] = result[c].interpolate(method="linear")

    after_nulls = int(result.isna().sum().sum())
    filled = before_nulls - after_nulls
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Interpolated ({method}): filled {filled:,} nulls ({before_nulls:,} → {after_nulls:,})",
    )
