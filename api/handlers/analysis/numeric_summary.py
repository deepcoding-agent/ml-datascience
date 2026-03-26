"""handle_numeric_summary handler."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_numeric_summary(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Comprehensive numeric columns summary in one table."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns found")

    rows: list[dict] = []
    for c in num_cols:
        s = df[c].dropna()
        rows.append({
            "column": c, "count": len(s),
            "null_pct": round(df[c].isnull().sum() / len(df) * 100, 1),
            "mean": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3), "p25": round(float(s.quantile(0.25)), 3),
            "median": round(float(s.median()), 3), "p75": round(float(s.quantile(0.75)), 3),
            "max": round(float(s.max()), 3), "skew": round(float(s.skew()), 3),
            "kurtosis": round(float(s.kurt()), 3),
            "zeros": int((s == 0).sum()), "negatives": int((s < 0).sum()),
        })

    result_df = pd.DataFrame(rows)

    heat_cols = ["mean", "std", "skew", "kurtosis"]
    heat_data = result_df.set_index("column")[heat_cols]
    fig = px.imshow(heat_data.T, text_auto=".2f", aspect="auto", color_continuous_scale="YlOrRd")
    _style(fig, title=f"Numeric Summary Heatmap — {len(num_cols)} columns")

    n_with_nulls = sum(1 for r in rows if r["null_pct"] > 0)
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Summarized {len(num_cols)} numeric columns. {n_with_nulls} have nulls.",
    )
