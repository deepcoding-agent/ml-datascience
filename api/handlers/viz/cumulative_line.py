"""handle_cumulative_line handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_cumulative_line(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Cumulative sum line chart."""
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
    if not col:
        return HandlerResult(success=False, error="No numeric column for cumulative line")
    cumsum = df[col].dropna().cumsum().reset_index(drop=True)
    fig = px.line(x=cumsum.index, y=cumsum.values)
    fig.update_traces(line_color="#FB8C3C")
    _style(fig, title=f"{col} — Cumulative Sum", xaxis_title="Index",
           yaxis_title=f"Cumulative {col}")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Cumulative line of '{col}'")
