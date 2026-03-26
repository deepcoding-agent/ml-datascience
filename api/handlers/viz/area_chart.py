"""handle_area_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_area_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Area chart for trends or compositions."""
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    y_col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
    fig = px.area(df.reset_index(), x="index", y=y_col)
    _style(fig, title=f"{y_col}")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Area chart of '{y_col}'")
