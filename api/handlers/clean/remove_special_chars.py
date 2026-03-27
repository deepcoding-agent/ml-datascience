"""handle_remove_special_chars handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_remove_special_chars(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Remove special characters, keep letters (all languages), digits, and spaces."""
    col = params.get("column")
    # \w matches Unicode word characters (letters + digits + underscore in all languages)
    # This keeps Thai, Chinese, Japanese, Korean, Arabic, etc.
    remove_pattern = params.get("pattern", r"[^\w\s]")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    for c in cols:
        result[c] = result[c].astype(str).str.replace(remove_pattern, "", regex=True).str.strip()
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Removed special characters from {len(cols)} column(s), kept text in all languages",
    )
