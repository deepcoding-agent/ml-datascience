"""handle_qq_plot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_qq_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
    """QQ plot — check if data follows normal distribution."""
    from scipy import stats as sp_stats

    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
    if not col:
        return HandlerResult(success=False, error="No numeric column for QQ plot")

    data = df[col].dropna().values
    (theoretical, sample), (slope, intercept, _) = sp_stats.probplot(data, dist="norm")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical, y=sample, mode="markers",
                             marker=dict(color="#FB8C3C", size=5, opacity=0.7), name="Data"))
    line_x = [min(theoretical), max(theoretical)]
    line_y = [slope * x + intercept for x in line_x]
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
                             line=dict(color="#457B9D", dash="dash"), name="Normal"))
    _style(fig, title=f"QQ Plot: {col}",
           xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"QQ plot of '{col}'")
