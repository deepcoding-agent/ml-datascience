"""handle_drop_nulls handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_drop_nulls(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    result = df.copy()
    before = len(result)
    if col and col in result.columns:
        result = result.dropna(subset=[col])
        summary = f"Dropped {before - len(result):,} rows with null in '{col}'"
    else:
        threshold = params.get("threshold", 0.5)
        # Drop columns with >threshold nulls, then drop rows with any remaining nulls
        null_pct = result.isnull().mean()
        cols_to_drop = null_pct[null_pct > threshold].index.tolist()
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)
        result = result.dropna()
        summary = f"Dropped {len(cols_to_drop)} columns (>{threshold*100:.0f}% null), then {before - len(result):,} rows with remaining nulls"
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=summary, metadata={"rows_before": before, "rows_after": len(result)})
