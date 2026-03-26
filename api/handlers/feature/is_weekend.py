"""handle_is_weekend handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_is_weekend(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create binary weekend flag from datetime column."""
    col = params.get("column")
    result = df.copy()
    dt_cols = result.select_dtypes(include="datetime").columns.tolist()
    if col and col in result.columns and col not in dt_cols:
        try:
            result[col] = pd.to_datetime(result[col], errors="coerce", format="mixed")
            dt_cols = [col]
        except Exception:
            return HandlerResult(success=False, error=f"Cannot parse '{col}' as datetime")
    elif col and col in dt_cols:
        dt_cols = [col]
    if not dt_cols:
        return HandlerResult(success=False, error="No datetime column found")
    for c in dt_cols:
        result[f"{c}_is_weekend"] = result[c].dt.dayofweek.isin([5, 6]).astype(int)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created is_weekend flag for {len(dt_cols)} datetime column(s)",
    )
