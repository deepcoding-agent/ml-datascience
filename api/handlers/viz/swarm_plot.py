"""handle_swarm_plot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_swarm_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Jitter/swarm plot — better spread than strip plot."""
    col = params.get("column")
    group_col = params.get("group")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
    group_col = group_col if group_col and group_col in df.columns else (cat_cols[0] if cat_cols else None)
    sample = df.sample(min(len(df), 500), random_state=42)
    if group_col:
        fig = px.strip(sample, x=group_col, y=col, color=group_col,
                       stripmode="overlay")
    else:
        fig = px.strip(sample, y=col)
    fig.update_traces(jitter=0.4, marker=dict(size=4, opacity=0.5))
    _style(fig, title=f"{col} — Swarm" + (f" by {group_col}" if group_col else ""),
           showlegend=False)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Swarm plot of '{col}'")
