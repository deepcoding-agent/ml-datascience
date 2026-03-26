"""handle_pie_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_pie_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    cats = df.select_dtypes(include=["object", "category"]).columns
    col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
    pie_df = _group_pie(df, col)
    fig = px.pie(pie_df, names="category", values="count", hole=0.4)
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      marker=dict(line=dict(color="white", width=2)),
                      hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percent: %{percent}<extra></extra>")
    _style(fig, title=f"{col} — Proportion (n={len(df):,})", showlegend=True)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Pie chart of '{col}'")
