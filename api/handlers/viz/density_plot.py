"""handle_density_plot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_density_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
    """KDE density plot — smoother alternative to histogram."""
    col = params.get("column")
    group_col = params.get("group")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
    if not col:
        return HandlerResult(success=False, error="No numeric column for density plot")

    if group_col and group_col in df.columns:
        fig = px.histogram(df, x=col, color=group_col, histnorm="density",
                           marginal="rug", barmode="overlay", opacity=0.5)
    else:
        fig = px.histogram(df, x=col, histnorm="density", marginal="rug")
        fig.update_traces(marker_color="#FB8C3C", opacity=0.7)

    _style(fig, title=f"Density: {col}")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Density plot of '{col}'")
