"""handle_normalize_text_case handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_normalize_text_case(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Normalize text to title/upper/lower/sentence case."""
    col = params.get("column")
    case = params.get("case", "lower")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()

    for c in cols:
        if case == "upper":
            result[c] = result[c].str.upper()
        elif case == "title":
            result[c] = result[c].str.title()
        elif case == "sentence":
            result[c] = result[c].str.capitalize()
        else:  # lower
            result[c] = result[c].str.lower()

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Normalized {len(cols)} column(s) to {case} case",
    )
