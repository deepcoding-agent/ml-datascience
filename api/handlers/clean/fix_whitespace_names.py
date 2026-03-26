"""handle_fix_whitespace_names handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fix_whitespace_names(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Fix excess whitespace in name/text: ' John  Doe ' → 'John Doe'."""
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    for c in cols:
        result[c] = result[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Fixed whitespace in {len(cols)} column(s)",
    )
