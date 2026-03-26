"""handle_time_series handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_time_series(df: pd.DataFrame, params: dict) -> HandlerResult:
    dt_cols = df.select_dtypes(include="datetime").columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not dt_cols:
        return HandlerResult(success=False, error="No datetime columns found")
    x = dt_cols[0]
    y = params.get("column") or (num_cols[0] if num_cols else df.columns[1])
    fig = px.line(df.sort_values(x), x=x, y=y)
    fig.update_traces(line_color="#FB8C3C")
    _style(fig, title=f"{y} over time")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Time series: {y} over {x}")
