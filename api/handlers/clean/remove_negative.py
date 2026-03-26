"""handle_remove_negative handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_remove_negative(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Remove rows with negative values in numeric columns."""
    col = params.get("column")
    result = df.copy()
    before = len(result)
    cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
    mask = result[cols].lt(0).any(axis=1)
    result = result[~mask].reset_index(drop=True)
    removed = before - len(result)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Removed {removed:,} rows with negative values ({before:,} → {len(result):,})",
    )
