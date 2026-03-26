"""handle_bottom_n_analysis handler."""
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


def handle_bottom_n_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Analyze bottom N rows by a metric."""
    col = params.get("column")
    n = int(params.get("n", 10))
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not col or col not in num_cols:
        col = num_cols[0] if num_cols else None
    if col is None:
        return HandlerResult(success=False, error="No numeric column found")

    bottom = df.nsmallest(n, col)
    overall_mean = float(df[col].mean())

    fig = px.bar(bottom.reset_index(drop=True), y=col, text=col)
    fig.update_traces(marker_color="#2EC4B6", texttemplate="%{text:,.2f}", textposition="outside")
    _style(fig, title=f"Bottom {n} by {col}")

    return HandlerResult(
        success=True, result_df=bottom, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Bottom {n} by '{col}': range [{bottom[col].min():,.2f}, {bottom[col].max():,.2f}], mean={bottom[col].mean():,.2f} (overall avg={overall_mean:,.2f})",
    )
