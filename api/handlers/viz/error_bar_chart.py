"""handle_error_bar_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_error_bar_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Bar chart with standard deviation error bars."""
    col = params.get("column")
    group_col = params.get("group")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
    group_col = group_col if group_col and group_col in df.columns else (cat_cols[0] if cat_cols else None)
    if not col or not group_col:
        return HandlerResult(success=False, error="Need a numeric column and a categorical column for error bars")
    agg = df.groupby(group_col)[col].agg(["mean", "std"]).reset_index()
    fig = go.Figure(go.Bar(x=agg[group_col], y=agg["mean"],
                           error_y=dict(type="data", array=agg["std"].fillna(0)),
                           marker_color="#FB8C3C"))
    _style(fig, title=f"{col} by {group_col} (mean +/- std)",
           xaxis_title=group_col, yaxis_title=col, bargap=0.3)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Error bar chart: {col} by {group_col}")
