"""handle_contour_plot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_contour_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Contour density plot of two numeric columns."""
    cols = params.get("columns", [])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    x = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
    y = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
    if not x or not y:
        return HandlerResult(success=False, error="Need at least 2 numeric columns for contour plot")
    fig = px.density_contour(df, x=x, y=y, marginal_x="histogram", marginal_y="histogram")
    fig.update_traces(contours_coloring="fill", colorscale="Oranges")
    _style(fig, title=f"{x} vs {y} — Contour Density")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Contour plot: {x} vs {y}")
