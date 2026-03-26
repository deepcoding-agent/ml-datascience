"""handle_drop_id_columns handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_drop_id_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Auto-detect and drop ID-like columns (high unique ratio, sequential, generic names)."""
    result = df.copy()
    id_cols = []

    for c in result.columns:
        col_lower = c.lower().strip("_")
        n_unique = result[c].nunique()
        unique_ratio = n_unique / len(result) if len(result) > 0 else 0

        # Name-based: short generic ID names
        is_id_name = (
            col_lower in ("id", "index", "idx", "key", "row", "num", "number", "seq", "serial")
            or (col_lower.endswith("id") and len(col_lower) <= 6)
            or col_lower == "unnamed: 0"
        )
        # Value-based: nearly all unique + numeric + sequential
        is_sequential = False
        if pd.api.types.is_numeric_dtype(result[c]) and unique_ratio > 0.95:
            sorted_vals = result[c].dropna().sort_values()
            if len(sorted_vals) > 1:
                diffs = sorted_vals.diff().dropna()
                is_sequential = (diffs == diffs.iloc[0]).mean() > 0.95

        if is_id_name or (unique_ratio > 0.95 and is_sequential):
            id_cols.append(c)

    if id_cols:
        result = result.drop(columns=id_cols)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Dropped {len(id_cols)} ID-like column(s): {id_cols}" if id_cols else "No ID-like columns detected",
    )
