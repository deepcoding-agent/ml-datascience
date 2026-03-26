"""handle_histogram handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_histogram(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    if not col or col not in df.columns:
        num = df.select_dtypes(include="number").columns
        col = num[0] if len(num) > 0 else df.columns[0]
    data = df[col].dropna()
    fig = px.histogram(df, x=col, marginal="box")
    fig.update_traces(marker_color="#FB8C3C", opacity=0.85)
    _style(fig, title=f"Distribution of {col} (n={len(data):,}, mean={data.mean():.2f})",
           xaxis_title=col, yaxis_title="Frequency")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Histogram of '{col}'")
