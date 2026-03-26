"""handle_waterfall_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_waterfall_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Waterfall chart — show incremental changes."""
    col = params.get("column")
    cat_col = params.get("category")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
    cat_col = cat_col if cat_col and cat_col in df.columns else (cat_cols[0] if cat_cols else None)
    if cat_col:
        agg = df.groupby(cat_col)[col].sum().head(10)
        labels, values = agg.index.tolist(), agg.values.tolist()
    else:
        data = df[col].dropna().head(15)
        labels = [str(i) for i in data.index]
        values = data.tolist()
    fig = go.Figure(go.Waterfall(x=labels, y=values,
                                 connector=dict(line=dict(color="#86868B")),
                                 increasing=dict(marker_color="#2EC4B6"),
                                 decreasing=dict(marker_color="#E71D36"),
                                 totals=dict(marker_color="#FB8C3C")))
    _style(fig, title=f"{col} — Waterfall", yaxis_title=col)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Waterfall chart of '{col}'")
