"""handle_standardize_dates handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_standardize_dates(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Parse and standardize mixed date formats to consistent datetime."""
    col = params.get("column")
    date_format = params.get("format", "%Y-%m-%d")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    converted: list[str] = []

    for c in cols:
        sample = result[c].dropna().head(100)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.7:
                result[c] = pd.to_datetime(result[c], errors="coerce", format="mixed")
                result[c] = result[c].dt.strftime(date_format).replace("NaT", None)
                converted.append(c)
        except Exception:
            pass

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Standardized dates in {len(converted)} column(s) to {date_format}" if converted else "No date columns detected",
    )
