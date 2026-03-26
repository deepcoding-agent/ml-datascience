"""handle_clip_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_clip_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Clip numeric values to percentile range (default 1st-99th)."""
    col = params.get("column")
    lower_pct = params.get("lower", 0.01)
    upper_pct = params.get("upper", 0.99)
    result = df.copy()
    cols = (
        [col] if col and col in result.columns
        else result.select_dtypes(include="number").columns.tolist()
    )
    clipped = []
    for c in cols:
        lo = result[c].quantile(lower_pct)
        hi = result[c].quantile(upper_pct)
        result[c] = result[c].clip(lo, hi)
        clipped.append(c)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Clipped {len(clipped)} columns to [{lower_pct:.0%}, {upper_pct:.0%}] percentile range",
    )
