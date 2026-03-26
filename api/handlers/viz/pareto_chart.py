"""handle_pareto_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_pareto_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Pareto chart — bar + cumulative line."""
    col = params.get("column")
    cats = df.select_dtypes(include=["object", "category"]).columns
    col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
    vc = df[col].value_counts().head(15).reset_index()
    vc.columns = [col, "count"]
    vc["cumulative_pct"] = (vc["count"].cumsum() / vc["count"].sum() * 100).round(1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=vc[col], y=vc["count"], name="Count",
                         marker_color="#FB8C3C"))
    fig.add_trace(go.Scatter(x=vc[col], y=vc["cumulative_pct"], name="Cumulative %",
                             yaxis="y2", mode="lines+markers",
                             line=dict(color="#457B9D", width=2)))
    fig.update_layout(yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                                  range=[0, 105], showgrid=False))
    _style(fig, title=f"{col} — Pareto", yaxis_title="Count", bargap=0.3)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Pareto chart of '{col}'")
