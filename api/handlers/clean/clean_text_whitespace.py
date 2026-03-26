"""handle_clean_text_whitespace handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_clean_text_whitespace(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Normalize all whitespace: double spaces, tabs, newlines → single space."""
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    for c in cols:
        result[c] = (
            result[c].astype(str)
            .str.replace(r"[\t\n\r]+", " ", regex=True)
            .str.replace(r"\s{2,}", " ", regex=True)
            .str.strip()
        )
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Normalized whitespace in {len(cols)} column(s)",
    )
