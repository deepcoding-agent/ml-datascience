"""handle_change_dtype handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_change_dtype(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Cast a column to a specific dtype (int, float, str, bool, datetime, category)."""
    col = params.get("column")
    dtype = params.get("dtype", "str")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    result = df.copy()
    try:
        if dtype in ("datetime", "date"):
            result[col] = pd.to_datetime(result[col], errors="coerce", format="mixed")
        elif dtype == "category":
            result[col] = result[col].astype("category")
        elif dtype == "bool":
            result[col] = result[col].astype(bool)
        elif dtype in ("int", "integer"):
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("Int64")
        elif dtype in ("float", "numeric"):
            result[col] = pd.to_numeric(result[col], errors="coerce")
        else:
            result[col] = result[col].astype(str)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Changed '{col}' dtype to {dtype}")
    except Exception as e:
        return HandlerResult(success=False, error=f"Cannot convert '{col}' to {dtype}: {e}")
