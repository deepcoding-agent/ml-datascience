"""handle_line_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_line_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    y_col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
    fig = px.line(df.reset_index(), x="index", y=y_col)
    fig.update_traces(line_color="#FB8C3C")
    _style(fig, title=f"{y_col} — Trend", xaxis_title="Index", yaxis_title=y_col)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Line chart of '{y_col}'")
