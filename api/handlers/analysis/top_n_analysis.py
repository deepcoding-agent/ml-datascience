"""handle_top_n_analysis handler."""
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


def handle_top_n_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Analyze top N rows by a metric with details."""
    col = params.get("column")
    n = int(params.get("n", 10))
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not col or col not in num_cols:
        col = num_cols[0] if num_cols else None
    if col is None:
        return HandlerResult(success=False, error="No numeric column found")

    top = df.nlargest(n, col)
    overall_mean = float(df[col].mean())
    top_mean = float(top[col].mean())

    fig = px.bar(top.reset_index(drop=True), y=col, text=col)
    fig.update_traces(marker_color="#FB8C3C", texttemplate="%{text:,.2f}", textposition="outside")
    _style(fig, title=f"Top {n} by {col}")

    ratio = top_mean / overall_mean if overall_mean != 0 else 0
    return HandlerResult(
        success=True, result_df=top, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Top {n} by '{col}': range [{top[col].min():,.2f}, {top[col].max():,.2f}], mean={top_mean:,.2f} ({ratio:.1f}x overall avg)",
    )
