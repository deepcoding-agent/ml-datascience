"""handle_fix_dtypes handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fix_dtypes(df: pd.DataFrame, params: dict) -> HandlerResult:
    result = df.copy()
    converted: dict[str, str] = {}
    for col in result.select_dtypes(include="object").columns:
        sample = result[col].dropna().head(100)
        if sample.empty:
            continue
        # Try numeric
        coerced = pd.to_numeric(sample, errors="coerce")
        if coerced.notna().mean() >= 0.8:
            result[col] = pd.to_numeric(result[col], errors="coerce")
            converted[col] = "numeric"
            continue
        # Try datetime
        try:
            coerced_dt = pd.to_datetime(sample, errors="coerce", format="mixed")
            if coerced_dt.notna().mean() >= 0.8:
                result[col] = pd.to_datetime(result[col], errors="coerce", format="mixed")
                converted[col] = "datetime"
        except Exception:
            pass
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Converted {len(converted)} columns: {converted}",
                         metadata={"converted": converted})
