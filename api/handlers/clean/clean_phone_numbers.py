"""handle_clean_phone_numbers handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_clean_phone_numbers(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Standardize phone numbers to digits-only format."""
    col = params.get("column")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    result = df.copy()
    result[col] = (
        result[col].astype(str)
        .str.replace(r"[^\d+]", "", regex=True)
        .str.strip("+")
    )
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Cleaned phone numbers in '{col}' to digits-only",
    )
