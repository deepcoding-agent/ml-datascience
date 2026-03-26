"""handle_box_plot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_box_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    if col and col in df.columns:
        data = df[col].dropna()
        fig = px.box(df, y=col, points="outliers")
        title = f"Box Plot: {col} (median={data.median():.2f})"
    else:
        num_cols = df.select_dtypes(include="number").columns[:6].tolist()
        melted = df[num_cols].melt(var_name="column", value_name="value")
        fig = px.box(melted, x="column", y="value", points="outliers")
        title = f"Box Plots — {len(num_cols)} Numeric Columns"
    _style(fig, title=title, yaxis_title="Value")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=title)
