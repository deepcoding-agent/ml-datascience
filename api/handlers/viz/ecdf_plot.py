"""handle_ecdf_plot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_ecdf_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Empirical cumulative distribution function plot."""
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns
    col = col if col and col in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
    if not col:
        return HandlerResult(success=False, error="No numeric column for ECDF")
    fig = px.ecdf(df, x=col)
    fig.update_traces(line_color="#FB8C3C")
    _style(fig, title=f"ECDF of {col}", xaxis_title=col, yaxis_title="Cumulative Probability")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"ECDF of '{col}'")
