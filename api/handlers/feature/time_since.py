"""handle_time_since handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_time_since(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Days since a reference date (default: first date in column)."""
    col = params.get("column")
    ref_date = params.get("reference")
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
        if ref_date:
            ref = pd.to_datetime(ref_date)
        else:
            ref = result[c].min()
        result[f"{c}_days_since"] = (result[c] - ref).dt.days
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created days_since features for {len(dt_cols)} column(s) (ref={ref_date or 'first date'})",
    )
