"""handle_split_name handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_split_name(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Split 'John Doe' into first_name and last_name columns."""
    col = params.get("column")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    result = df.copy()
    parts = result[col].astype(str).str.strip().str.split(r"\s+", n=1, expand=True)
    result["first_name"] = parts[0] if 0 in parts.columns else ""
    result["last_name"] = parts[1] if 1 in parts.columns else ""
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Split '{col}' into first_name and last_name columns",
    )
