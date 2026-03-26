"""handle_fill_forward_backward handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fill_forward_backward(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Fill nulls using forward fill then backward fill."""
    col = params.get("column")
    result = df.copy()
    before_nulls = int(result.isna().sum().sum())

    if col and col in result.columns:
        result[col] = result[col].ffill().bfill()
    else:
        result = result.ffill().bfill()

    after_nulls = int(result.isna().sum().sum())
    filled = before_nulls - after_nulls
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Forward+backward fill: resolved {filled:,} nulls ({before_nulls:,} → {after_nulls:,})",
    )
