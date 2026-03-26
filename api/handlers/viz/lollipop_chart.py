"""handle_lollipop_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_lollipop_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Lollipop chart — dot + stem line."""
    col = params.get("column")
    cats = df.select_dtypes(include=["object", "category"]).columns
    col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
    vc = df[col].value_counts().head(15).reset_index()
    vc.columns = [col, "count"]
    fig = go.Figure()
    for _, row in vc.iterrows():
        fig.add_trace(go.Scatter(x=[0, row["count"]], y=[row[col], row[col]],
                                 mode="lines", line=dict(color="#86868B", width=1.5),
                                 showlegend=False))
    fig.add_trace(go.Scatter(x=vc["count"], y=vc[col], mode="markers",
                             marker=dict(color="#FB8C3C", size=10),
                             name="Count"))
    _style(fig, title=f"{col} — Lollipop", xaxis_title="Count")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Lollipop chart of '{col}'")
