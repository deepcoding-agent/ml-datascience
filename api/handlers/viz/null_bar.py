"""handle_null_bar handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_null_bar(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Bar chart of null percentage per column."""
    null_pct = (df.isnull().sum() / len(df) * 100).round(2)
    null_pct = null_pct[null_pct > 0].sort_values(ascending=False)
    if len(null_pct) == 0:
        return HandlerResult(success=True, charts_plotly=[],
                             summary="No null values found in any column")
    ndf = null_pct.reset_index()
    ndf.columns = ["column", "null_pct"]
    fig = px.bar(ndf, x="column", y="null_pct", text="null_pct")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                      marker_color="#E71D36")
    _style(fig, title="Null Percentage per Column", xaxis_title="Column",
           yaxis_title="Null %", bargap=0.3)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Null bar: {len(null_pct)} columns with nulls")
