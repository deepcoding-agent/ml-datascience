"""handle_strip_plot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_strip_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Strip/jitter plot — shows individual data points."""
    col = params.get("column")
    group_col = params.get("group")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
    group_col = group_col if group_col and group_col in df.columns else (cat_cols[0] if cat_cols else None)

    if group_col:
        fig = px.strip(df, x=group_col, y=col, color=group_col)
    else:
        fig = px.strip(df, y=col)

    fig.update_traces(marker=dict(size=5, opacity=0.6))
    _style(fig, title=f"{col}" + (f" by {group_col}" if group_col else ""), showlegend=False)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Strip plot of '{col}'")
