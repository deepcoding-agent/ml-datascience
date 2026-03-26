"""handle_is_holiday handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_is_holiday(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Basic holiday detection — weekends + common fixed-date holidays."""
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
    common_holidays = {(1, 1), (7, 4), (12, 25), (12, 31), (1, 26), (5, 1)}
    for c in dt_cols:
        is_wknd = result[c].dt.dayofweek.isin([5, 6])
        month_day = list(zip(result[c].dt.month, result[c].dt.day))
        is_fixed = pd.Series([md in common_holidays for md in month_day], index=result.index)
        result[f"{c}_is_holiday"] = (is_wknd | is_fixed).astype(int)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created is_holiday flag for {len(dt_cols)} datetime column(s)",
    )
