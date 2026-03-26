"""handle_clean_currency handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_clean_currency(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Clean currency strings ($, EUR, ¥, commas) to float."""
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    converted: list[str] = []

    currency_re = re.compile(r"[$€£¥₹₩₫฿,\s]")
    for c in cols:
        sample = result[c].dropna().head(100).astype(str)
        if sample.empty:
            continue
        has_currency = sample.str.contains(r"[$€£¥₹₩₫฿]", regex=True).any()
        if not has_currency:
            continue
        cleaned = result[c].astype(str).apply(lambda v: currency_re.sub("", v))
        coerced = pd.to_numeric(cleaned, errors="coerce")
        if coerced.notna().mean() >= 0.5:
            result[c] = coerced
            converted.append(c)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Cleaned currency in {len(converted)} column(s): {converted}" if converted else "No currency columns detected",
    )
