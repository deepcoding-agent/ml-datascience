"""handle_gauge_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_gauge_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Single-value gauge chart — shows a key metric."""
    col = params.get("column")
    agg = params.get("agg", "mean")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
    if not col:
        return HandlerResult(success=False, error="No numeric column for gauge")
    data = df[col].dropna()
    agg_funcs = {"mean": data.mean, "median": data.median, "sum": data.sum,
                 "min": data.min, "max": data.max}
    value = float(agg_funcs.get(agg, data.mean)())
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=f"{col} ({agg})"),
        gauge=dict(axis=dict(range=[float(data.min()), float(data.max())]),
                   bar=dict(color="#FB8C3C"),
                   steps=[
                       dict(range=[float(data.min()), float(data.quantile(0.5))], color="#F0F0F0"),
                       dict(range=[float(data.quantile(0.5)), float(data.max())], color="#E8E8E8")
                   ])))
    _style(fig, title=f"{col} — Gauge ({agg}={value:,.2f})")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Gauge: {col} {agg}={value:,.2f}")
