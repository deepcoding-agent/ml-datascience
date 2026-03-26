"""handle_fix_boolean handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fix_boolean(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Standardize boolean-like values (yes/no, true/false, Y/N, 1/0) to bool."""
    col = params.get("column")
    result = df.copy()
    true_vals = {"yes", "y", "true", "t", "1", "1.0", "si", "on"}
    false_vals = {"no", "n", "false", "f", "0", "0.0", "off"}
    converted: list[str] = []

    cols = [col] if col and col in result.columns else result.select_dtypes(include="object").columns.tolist()
    for c in cols:
        unique = result[c].dropna().astype(str).str.strip().str.lower().unique()
        all_bool = all(v in true_vals | false_vals for v in unique)
        if all_bool and len(unique) > 0:
            result[c] = result[c].astype(str).str.strip().str.lower().map(
                lambda v, t=true_vals: True if v in t else (False if v in false_vals else None)
            )
            converted.append(c)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Converted {len(converted)} column(s) to boolean: {converted}" if converted else "No boolean-like columns detected",
    )
