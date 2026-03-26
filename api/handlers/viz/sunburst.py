"""handle_sunburst handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_sunburst(df: pd.DataFrame, params: dict) -> HandlerResult:
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cats) < 2:
        return HandlerResult(success=False, error="Need at least 2 categorical columns for sunburst")
    fig = px.sunburst(df, path=cats[:2])
    _style(fig, title=f"{cats[0]} → {cats[1]}")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Sunburst: {cats[0]} → {cats[1]}")
