"""handle_radar_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_radar_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Radar/spider chart — compare numeric columns for a row or aggregation."""
    num_cols = df.select_dtypes(include="number").columns[:8].tolist()
    if len(num_cols) < 3:
        return HandlerResult(success=False, error="Need at least 3 numeric columns for radar chart")
    means = df[num_cols].mean()
    normalized = ((means - means.min()) / (means.max() - means.min()) * 100).round(1)
    fig = go.Figure(go.Scatterpolar(
        r=normalized.tolist() + [normalized.tolist()[0]],
        theta=num_cols + [num_cols[0]],
        fill="toself", fillcolor="rgba(251,140,60,0.3)",
        line=dict(color="#FB8C3C")))
    _style(fig, title="Radar — Normalized Means")
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Radar chart of {len(num_cols)} columns")
