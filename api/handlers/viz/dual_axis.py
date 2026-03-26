"""handle_dual_axis handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_dual_axis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Dual Y-axis chart — two numeric columns on separate axes."""
    cols = params.get("columns", [])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    y1 = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
    y2 = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
    if not y1 or not y2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns for dual axis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df[y1], mode="lines", name=y1,
                             line=dict(color="#FB8C3C")))
    fig.add_trace(go.Scatter(y=df[y2], mode="lines", name=y2,
                             yaxis="y2", line=dict(color="#457B9D")))
    fig.update_layout(yaxis2=dict(title=y2, overlaying="y", side="right",
                                  showgrid=False))
    _style(fig, title=f"{y1} & {y2} — Dual Axis", yaxis_title=y1)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Dual axis: {y1} vs {y2}")
