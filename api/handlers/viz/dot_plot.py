"""handle_dot_plot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_dot_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Cleveland dot plot — clean comparison of values across categories."""
    col = params.get("column")
    value_col = params.get("value")
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in df.columns else (cats[0] if cats else df.columns[0])
    value_col = value_col if value_col and value_col in df.columns else (num_cols[0] if num_cols else None)
    if value_col:
        agg = df.groupby(col)[value_col].mean().sort_values().tail(15).reset_index()
        agg.columns = [col, "value"]
    else:
        agg = df[col].value_counts().tail(15).reset_index()
        agg.columns = [col, "value"]
    fig = go.Figure(go.Scatter(x=agg["value"], y=agg[col], mode="markers",
                               marker=dict(color="#FB8C3C", size=10)))
    _style(fig, title=f"{col} — Dot Plot", xaxis_title=value_col or "Count")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Dot plot of '{col}'")
