"""handle_time_series_decompose handler."""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger
log = get_logger(__name__)

def handle_time_series_decompose(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Decompose time series into trend, seasonal, and residual components."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    col = params.get("column")
    period = params.get("period")
    nums = BaseHandler.get_numeric_cols(df)
    if not col:
        col = nums[0] if nums else None
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Numeric column required. Available: {nums}")
    series = df[col].dropna()
    if len(series) < 10:
        return HandlerResult(success=False, error="Need at least 10 data points")
    if not period:
        period = min(max(len(series) // 4, 2), 52)
    period = int(period)
    try:
        decomp = seasonal_decompose(series, model="additive", period=period)
    except Exception as e:
        return HandlerResult(success=False, error=f"Decomposition failed: {e}")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    x = list(range(len(series)))
    fig.add_trace(go.Scatter(x=x, y=series.values, mode="lines", name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=decomp.trend, mode="lines", name="Trend", line=dict(color="#FB8C3C")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=decomp.seasonal, mode="lines", name="Seasonal", line=dict(color="#2EC4B6")), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=decomp.resid, mode="lines", name="Residual", line=dict(color="#457B9D")), row=4, col=1)
    _style(fig, title=f"Time Series Decomposition: {col} (period={period})", height=700)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()], output_type="query",
                         summary=f"Decomposed \'{col}\' (period={period}): trend + seasonal + residual")
