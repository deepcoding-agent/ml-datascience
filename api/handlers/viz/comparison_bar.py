"""handle_comparison_bar handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_comparison_bar(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Before/after or A vs B comparison bar chart."""
    cols = params.get("columns", [])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    a = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
    b = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
    if not a or not b:
        return HandlerResult(success=False, error="Need 2 numeric columns for comparison")
    stats = ["mean", "median", "std", "min", "max"]
    vals_a = [df[a].mean(), df[a].median(), df[a].std(), df[a].min(), df[a].max()]
    vals_b = [df[b].mean(), df[b].median(), df[b].std(), df[b].min(), df[b].max()]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=stats, y=vals_a, name=a, marker_color="#FB8C3C"))
    fig.add_trace(go.Bar(x=stats, y=vals_b, name=b, marker_color="#457B9D"))
    fig.update_layout(barmode="group")
    _style(fig, title=f"{a} vs {b} — Comparison", bargap=0.3)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Comparison: {a} vs {b}")
