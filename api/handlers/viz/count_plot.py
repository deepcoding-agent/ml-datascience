"""handle_count_plot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_count_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    cats = df.select_dtypes(include=["object", "category"]).columns
    col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
    vc = df[col].value_counts().head(20).reset_index()
    vc.columns = [col, "count"]
    total = len(df)
    vc["pct"] = (vc["count"] / total * 100).round(1)
    fig = px.bar(vc, x=col, y="count", text="count", hover_data={"pct": ":.1f"})
    fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                      marker_color="#FB8C3C",
                      hovertemplate=f"<b>%{{x}}</b><br>Count: %{{y:,}}<br>Percent: %{{customdata[0]:.1f}}%<extra></extra>")
    _style(fig, title=f"{col} — Frequency (n={total:,})",
           xaxis_title=col, yaxis_title="Count", bargap=0.3)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Count plot of '{col}'")
