"""handle_column_compare handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_column_compare(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compare two columns statistically."""
    cols = params.get("columns", [])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col1 = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
    col2 = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
    if not col1 or not col2:
        return HandlerResult(success=False, error="Need 2 numeric columns to compare")

    corr_val = float(df[[col1, col2]].corr().iloc[0, 1])
    result = pd.DataFrame([{
        "stat": s,
        col1: round(float(getattr(df[col1], s)()), 4) if s != "count" else int(df[col1].count()),
        col2: round(float(getattr(df[col2], s)()), 4) if s != "count" else int(df[col2].count()),
    } for s in ["count", "mean", "std", "min", "median", "max"]] + [{
        "stat": "correlation", col1: round(corr_val, 4), col2: round(corr_val, 4),
    }])
    return HandlerResult(success=True, result_df=result,
                         summary=f"Compare {col1} vs {col2}: corr={corr_val:.4f}")
