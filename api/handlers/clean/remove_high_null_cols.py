"""handle_remove_high_null_cols handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_remove_high_null_cols(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Drop columns above a null-percentage threshold."""
    threshold = params.get("threshold", 0.5)
    result = df.copy()
    null_pct = result.isnull().mean()
    cols_to_drop = null_pct[null_pct > threshold].index.tolist()
    if cols_to_drop:
        result = result.drop(columns=cols_to_drop)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Dropped {len(cols_to_drop)} column(s) with >{threshold*100:.0f}% nulls: {cols_to_drop}" if cols_to_drop else f"No columns exceed {threshold*100:.0f}% null threshold",
    )
