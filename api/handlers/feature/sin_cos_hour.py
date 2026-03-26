"""handle_sin_cos_hour handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_sin_cos_hour(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Sin/cos encoding for hour-of-day column (period=24)."""
    col = params.get("column")
    result = df.copy()
    if col and col in result.columns:
        hours = pd.to_numeric(result[col], errors="coerce")
    else:
        dt_cols = result.select_dtypes(include="datetime").columns.tolist()
        if not dt_cols:
            return HandlerResult(success=False, error="No datetime or hour column found")
        col = dt_cols[0]
        hours = result[col].dt.hour
    result[f"{col}_hour_sin"] = np.sin(2 * np.pi * hours / 24).round(4)
    result[f"{col}_hour_cos"] = np.cos(2 * np.pi * hours / 24).round(4)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Sin/cos hour encoding from '{col}' (period=24)",
    )
