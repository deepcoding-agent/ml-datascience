"""handle_ratio_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_ratio_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compute key ratios between numeric columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")
    rows = []
    for i in range(min(len(num_cols), 10)):
        for j in range(i + 1, min(len(num_cols), 10)):
            a, b = num_cols[i], num_cols[j]
            mean_a, mean_b = float(df[a].mean()), float(df[b].mean())
            ratio = round(mean_a / mean_b, 4) if mean_b != 0 else float("inf")
            rows.append({"col_a": a, "col_b": b, "mean_a": round(mean_a, 4),
                         "mean_b": round(mean_b, 4), "ratio": ratio})
    result = pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result,
                         summary=f"Mean ratios between {min(len(num_cols), 10)} numeric columns")
