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
    """Remove special characters, keep alphanumeric + spaces."""
    col = params.get("column")
    keep_pattern = params.get("keep", r"[^a-zA-Z0-9\s]")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    for c in cols:
        result[c] = result[c].astype(str).str.replace(keep_pattern, "", regex=True)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Removed special characters from {len(cols)} column(s)",
    )
