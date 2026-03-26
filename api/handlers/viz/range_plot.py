"""handle_range_plot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_range_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Range/band plot showing min-max range over categories."""
    col = params.get("column")
    group_col = params.get("group")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
    group_col = group_col if group_col and group_col in df.columns else (cat_cols[0] if cat_cols else None)
    if not col or not group_col:
        return HandlerResult(success=False, error="Need a numeric and a categorical column for range plot")
    agg = df.groupby(group_col)[col].agg(["min", "max", "mean"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg[group_col], y=agg["max"], mode="lines",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=agg[group_col], y=agg["min"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(251,140,60,0.25)", name="Min-Max Range"))
    fig.add_trace(go.Scatter(x=agg[group_col], y=agg["mean"], mode="lines+markers",
                             line=dict(color="#FB8C3C", width=2), name="Mean"))
    _style(fig, title=f"{col} Range by {group_col}", yaxis_title=col)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Range plot: {col} by {group_col}")
