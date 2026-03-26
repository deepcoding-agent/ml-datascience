"""handle_scatter handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_scatter(df: pd.DataFrame, params: dict) -> HandlerResult:
    cols = params.get("columns", [])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    x = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
    y = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else df.columns[-1])
    color = cols[2] if len(cols) > 2 and cols[2] in df.columns else None
    fig = px.scatter(df, x=x, y=y, color=color, opacity=0.7,
                     trendline="ols" if color is None else None)
    _style(fig, title=f"{y} vs {x} (n={len(df):,})",
           xaxis_title=x, yaxis_title=y)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Scatter plot: {x} vs {y}")
