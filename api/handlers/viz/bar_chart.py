"""handle_bar_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_bar_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    cats = df.select_dtypes(include=["object", "category"]).columns
    col = params.get("column") or (cats[0] if len(cats) > 0 else df.columns[0])
    show_pct = params.get("percentage", False)
    vc = df[col].value_counts().head(15).reset_index()
    vc.columns = [col, "count"]
    total = len(df)
    vc["pct"] = (vc["count"] / total * 100).round(1)
    if show_pct:
        fig = px.bar(vc, x=col, y="pct", text="pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          marker_color="#FB8C3C")
        _style(fig, title=f"{col} — Percentage Distribution",
               xaxis_title=col, yaxis_title="Percentage (%)", bargap=0.3)
    else:
        fig = px.bar(vc, x=col, y="count", text="count",
                     hover_data={"pct": ":.1f"})
        fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                          marker_color="#FB8C3C",
                          hovertemplate=f"<b>%{{x}}</b><br>Count: %{{y:,}}<br>Percent: %{{customdata[0]:.1f}}%<extra></extra>")
        _style(fig, title=f"{col} — Value Counts (n={total:,})",
               xaxis_title=col, yaxis_title="Count", bargap=0.3)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Bar chart of '{col}'")
