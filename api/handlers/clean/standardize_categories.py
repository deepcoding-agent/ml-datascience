"""handle_standardize_categories handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_standardize_categories(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Merge similar categories by stripping, lowering, and mapping common variants."""
    col = params.get("column")
    mapping = params.get("mapping", {})
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    result = df.copy()
    before_unique = result[col].nunique()
    # Strip + lowercase
    result[col] = result[col].astype(str).str.strip().str.lower()
    # Apply explicit mapping if provided
    if mapping:
        result[col] = result[col].replace(mapping)
    after_unique = result[col].nunique()
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Standardized '{col}' categories: {before_unique} → {after_unique} unique values",
    )
