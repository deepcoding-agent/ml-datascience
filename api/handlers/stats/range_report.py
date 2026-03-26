"""handle_range_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_range_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Range (max - min) per numeric column."""
    num_cols = df.select_dtypes(include="number").columns
    rows = []
    for c in num_cols:
        mn, mx = float(df[c].min()), float(df[c].max())
        rows.append({"column": c, "min": round(mn, 4), "max": round(mx, 4), "range": round(mx - mn, 4)})
    result = pd.DataFrame(rows).sort_values("range", ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, summary="Range per numeric column")
