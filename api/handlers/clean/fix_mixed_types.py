"""handle_fix_mixed_types handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fix_mixed_types(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Convert mixed-type columns to a consistent type (numeric preferred, else string)."""
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.columns.tolist()
    converted: dict[str, str] = {}

    for c in cols:
        if result[c].apply(type).nunique() <= 1:
            continue
        # Try numeric first
        coerced = pd.to_numeric(result[c], errors="coerce")
        if coerced.notna().mean() >= 0.7:
            result[c] = coerced
            converted[c] = "numeric"
        else:
            result[c] = result[c].astype(str)
            converted[c] = "string"

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Fixed mixed types in {len(converted)} column(s): {converted}" if converted else "No mixed-type columns found",
        metadata={"converted": converted},
    )
