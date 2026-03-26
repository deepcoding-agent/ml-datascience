"""handle_distribution handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_distribution(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns
    col = col if col and col in df.columns else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
    data = df[col].dropna()
    fig = px.histogram(df, x=col, marginal="box")
    fig.update_traces(marker_color="#FB8C3C", opacity=0.85)
    _style(fig, title=f"Distribution of {col} (n={len(data):,}, mean={data.mean():.2f}, std={data.std():.2f})",
           xaxis_title=col, yaxis_title="Frequency")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Distribution of '{col}'")
