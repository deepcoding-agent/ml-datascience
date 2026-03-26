"""handle_deduplicate_by handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_deduplicate_by(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Remove duplicates based on specific column(s), keeping first or last."""
    cols = params.get("columns", [])
    col = params.get("column")
    keep = params.get("keep", "first")

    subset = cols if cols else ([col] if col and col in df.columns else None)
    if subset:
        subset = [c for c in subset if c in df.columns]
        if not subset:
            return HandlerResult(success=False, error="No valid columns specified for deduplication")

    before = len(df)
    result = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    removed = before - len(result)
    col_desc = f" by {subset}" if subset else ""
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Removed {removed:,} duplicates{col_desc} (keep={keep}): {before:,} → {len(result):,}",
    )
