"""handle_top_n_bar handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_top_n_bar(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Top N values bar chart."""
    col = params.get("column")
    n = params.get("n", 10)
    cats = df.select_dtypes(include=["object", "category"]).columns
    col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
    vc = df[col].value_counts().head(n).reset_index()
    vc.columns = [col, "count"]
    fig = px.bar(vc, x=col, y="count", text="count")
    fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                      marker_color="#FB8C3C")
    _style(fig, title=f"Top {n} — {col}", xaxis_title=col,
           yaxis_title="Count", bargap=0.3)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Top {n} bar chart of '{col}'")
