"""handle_parallel_coords handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_parallel_coords(df: pd.DataFrame, params: dict) -> HandlerResult:
    num_cols = df.select_dtypes(include="number").columns[:6].tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")
    fig = px.parallel_coordinates(df[num_cols].dropna(), dimensions=num_cols)
    _style(fig, title="Parallel Coordinates")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary="Parallel coordinates plot")
