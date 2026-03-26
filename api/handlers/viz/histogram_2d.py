"""handle_histogram_2d handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_histogram_2d(df: pd.DataFrame, params: dict) -> HandlerResult:
    """2D histogram / density heatmap of two numeric columns."""
    cols = params.get("columns", [])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    x = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
    y = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
    if not x or not y:
        return HandlerResult(success=False, error="Need at least 2 numeric columns for 2D histogram")
    fig = px.density_heatmap(df, x=x, y=y, marginal_x="histogram", marginal_y="histogram",
                             color_continuous_scale="Oranges")
    _style(fig, title=f"{x} vs {y} — 2D Density")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"2D histogram: {x} vs {y}")
