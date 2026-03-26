"""handle_heatmap handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_heatmap(df: pd.DataFrame, params: dict) -> HandlerResult:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns for heatmap")
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlBu_r",
                    aspect="auto")
    _style(fig, title="Correlation Heatmap")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary="Correlation heatmap")
