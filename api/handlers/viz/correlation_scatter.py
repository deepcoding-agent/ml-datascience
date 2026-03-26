"""handle_correlation_scatter handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_correlation_scatter(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Scatter plot with OLS trendline and R-squared value."""
    cols = params.get("columns", [])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    x = cols[0] if len(cols) > 0 and cols[0] in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
    y = cols[1] if len(cols) > 1 and cols[1] in df.columns else (num_cols[1] if len(num_cols) > 1 else None)
    if not x or not y:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")
    clean = df[[x, y]].dropna()
    r = clean[x].corr(clean[y])
    fig = px.scatter(clean, x=x, y=y, trendline="ols", opacity=0.6)
    fig.update_traces(marker_color="#FB8C3C", selector=dict(mode="markers"))
    _style(fig, title=f"{x} vs {y} (r = {r:.3f})",
           xaxis_title=x, yaxis_title=y)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Correlation scatter: {x} vs {y}, r={r:.3f}")
