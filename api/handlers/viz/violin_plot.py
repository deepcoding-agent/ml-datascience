"""handle_violin_plot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_violin_plot(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns
    col = col if col and col in df.columns else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
    data = df[col].dropna()
    fig = px.violin(df, y=col, box=True, points="outliers")
    _style(fig, title=f"Violin: {col} (median={data.median():.2f})",
           yaxis_title=col)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Violin plot of '{col}'")
