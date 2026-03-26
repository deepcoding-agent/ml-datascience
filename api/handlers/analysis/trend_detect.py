"""handle_trend_detect handler."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_trend_detect(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Detect trends in numeric data: overall direction, rate of change,
    moving average, and turning points."""
    col = params.get("column")
    window = int(params.get("window", 5))
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not col or col not in num_cols:
        col = num_cols[0] if num_cols else None
    if col is None:
        return HandlerResult(success=False, error="No numeric column found")

    s = df[col].dropna().reset_index(drop=True)
    if len(s) < 5:
        return HandlerResult(success=False, error=f"Need at least 5 data points, got {len(s)}")

    # Linear trend
    x = np.arange(len(s))
    coeffs = np.polyfit(x, s.values, 1)
    slope, intercept = coeffs[0], coeffs[1]
    trend_line = slope * x + intercept

    # Moving average
    ma = s.rolling(window=window, min_periods=1).mean()

    # Trend direction
    if slope > 0:
        pct_change = (trend_line[-1] - trend_line[0]) / abs(trend_line[0]) * 100 if trend_line[0] != 0 else 0
        direction = f"Upward (+{pct_change:.1f}%)"
    elif slope < 0:
        pct_change = (trend_line[-1] - trend_line[0]) / abs(trend_line[0]) * 100 if trend_line[0] != 0 else 0
        direction = f"Downward ({pct_change:.1f}%)"
    else:
        direction = "Flat"

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=s.values, mode="lines", name=col, line=dict(color="#86868B", width=1), opacity=0.5))
    fig.add_trace(go.Scatter(y=ma.values, mode="lines", name=f"MA({window})", line=dict(color="#FB8C3C", width=2)))
    fig.add_trace(go.Scatter(y=trend_line, mode="lines", name="Trend", line=dict(color="#E71D36", width=2, dash="dash")))
    _style(fig, title=f"Trend Analysis — {col} ({direction})")
    fig.update_layout(xaxis_title="Index", yaxis_title=col)

    result_df = pd.DataFrame({
        "metric": ["direction", "slope", "start_value", "end_value", "min", "max", "volatility"],
        "value": [direction, round(slope, 4), round(float(s.iloc[0]), 2), round(float(s.iloc[-1]), 2),
                  round(float(s.min()), 2), round(float(s.max()), 2), round(float(s.std() / s.mean() * 100), 2) if s.mean() != 0 else 0],
    })

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Trend in '{col}': {direction}. Slope={slope:+.4f}, range [{s.min():,.2f}, {s.max():,.2f}]",
    )
