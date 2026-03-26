"""handle_remove_rare_categories handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_remove_rare_categories(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Replace categories with fewer than N occurrences with 'Other'."""
    col = params.get("column")
    min_count = params.get("min_count", 5)
    replacement = params.get("replacement", "Other")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    result = df.copy()
    counts = result[col].value_counts()
    rare = counts[counts < min_count].index.tolist()
    result[col] = result[col].replace(rare, replacement)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Replaced {len(rare)} rare categories (count < {min_count}) with '{replacement}' in '{col}'",
    )
