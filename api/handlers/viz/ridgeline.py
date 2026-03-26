"""handle_ridgeline handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_ridgeline(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Ridgeline plot — overlapping distributions by group."""
    col = params.get("column")
    group_col = params.get("group")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
    group_col = group_col if group_col and group_col in df.columns else (cat_cols[0] if cat_cols else None)
    if not col or not group_col:
        return HandlerResult(success=False, error="Need a numeric column and a group column for ridgeline")
    groups = df[group_col].value_counts().head(8).index.tolist()
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, grp in enumerate(groups):
        data = df[df[group_col] == grp][col].dropna()
        fig.add_trace(go.Violin(x=data, name=str(grp), side="positive",
                                line_color=colors[i % len(colors)], meanline_visible=True))
    fig.update_traces(orientation="h", width=1.8)
    _style(fig, title=f"{col} by {group_col} — Ridgeline", showlegend=True)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Ridgeline: {col} by {group_col}")
