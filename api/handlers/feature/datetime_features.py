"""handle_datetime_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_datetime_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract year, month, day, dayofweek, hour from datetime columns."""
    col = params.get("column")
    result = df.copy()

    # Find datetime columns
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
        return HandlerResult(success=False, error="No datetime columns found")

    created = []
    for c in dt_cols:
        dt = result[c].dt
        result[f"{c}_year"] = dt.year
        result[f"{c}_month"] = dt.month
        result[f"{c}_day"] = dt.day
        result[f"{c}_dayofweek"] = dt.dayofweek
        if dt.hour.max() > 0:
            result[f"{c}_hour"] = dt.hour
            created.extend([f"{c}_year", f"{c}_month", f"{c}_day", f"{c}_dayofweek", f"{c}_hour"])
        else:
            created.extend([f"{c}_year", f"{c}_month", f"{c}_day", f"{c}_dayofweek"])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Extracted {len(created)} datetime features from {len(dt_cols)} column(s)",
    )
