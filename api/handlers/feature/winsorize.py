"""handle_winsorize handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_winsorize(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Cap extreme values at percentile bounds (default 5th-95th)."""
    col = params.get("column")
    lower_pct = params.get("lower", 0.05)
    upper_pct = params.get("upper", 0.95)
    result = df.copy()
    cols = (
        [col] if col and col in result.columns
        else result.select_dtypes(include="number").columns.tolist()
    )
    winsorized = []
    for c in cols:
        lo = result[c].quantile(lower_pct)
        hi = result[c].quantile(upper_pct)
        result[c] = result[c].clip(lo, hi)
        winsorized.append(c)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Winsorized {len(winsorized)} columns to [{lower_pct:.0%}, {upper_pct:.0%}]",
    )
