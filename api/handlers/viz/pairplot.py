"""handle_pairplot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_pairplot(df: pd.DataFrame, params: dict) -> HandlerResult:
    num_cols = df.select_dtypes(include="number").columns[:5].tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns for pairplot")
    fig = px.scatter_matrix(df[num_cols].dropna(), dimensions=num_cols)
    fig.update_traces(diagonal_visible=True, marker=dict(size=3, opacity=0.5))
    _style(fig, title="Pair Plot")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Pair plot of {len(num_cols)} columns")
