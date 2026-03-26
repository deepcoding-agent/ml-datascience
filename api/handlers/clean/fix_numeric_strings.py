"""handle_fix_numeric_strings handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fix_numeric_strings(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Convert formatted numeric strings ('$1,234' / '1.234,56') to float."""
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    converted: list[str] = []

    for c in cols:
        sample = result[c].dropna().head(100).astype(str)
        if sample.empty:
            continue
        cleaned = sample.str.replace(r"[^\d.,\-]", "", regex=True)
        # Detect European format (1.234,56) vs US format (1,234.56)
        has_european = cleaned.str.contains(r"\d\.\d{3},", regex=True).any()
        if has_european:
            cleaned = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        else:
            cleaned = cleaned.str.replace(",", "", regex=False)
        coerced = pd.to_numeric(cleaned, errors="coerce")
        if coerced.notna().mean() >= 0.6:
            full = result[c].astype(str).str.replace(r"[^\d.,\-]", "", regex=True)
            if has_european:
                full = full.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            else:
                full = full.str.replace(",", "", regex=False)
            result[c] = pd.to_numeric(full, errors="coerce")
            converted.append(c)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Converted {len(converted)} column(s) from formatted strings to numeric: {converted}" if converted else "No numeric-string columns detected",
    )
