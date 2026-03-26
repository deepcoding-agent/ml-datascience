"""handle_bubble_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_bubble_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 3:
        return HandlerResult(success=False, error="Need at least 3 numeric columns for bubble chart")
    fig = px.scatter(df, x=num_cols[0], y=num_cols[1], size=num_cols[2],
                     opacity=0.7)
    _style(fig, title=f"{num_cols[0]} vs {num_cols[1]} (size: {num_cols[2]})")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary="Bubble chart")
