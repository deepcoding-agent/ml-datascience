"""handle_dedup_keep_latest handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_dedup_keep_latest(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Deduplicate by a column, keeping the row with the latest value in another column."""
    key_col = params.get("column") or params.get("key_column")
    date_col = params.get("date_column") or params.get("sort_column")
    if not key_col or key_col not in df.columns:
        return HandlerResult(success=False, error=f"Key column '{key_col}' not found")
    if not date_col or date_col not in df.columns:
        return HandlerResult(success=False, error=f"Date/sort column '{date_col}' not found — needed to determine 'latest'")
    result = df.sort_values(date_col, ascending=True).drop_duplicates(subset=[key_col], keep="last").reset_index(drop=True)
    removed = len(df) - len(result)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Deduped by '{key_col}' keeping latest by '{date_col}': removed {removed:,} rows",
    )
