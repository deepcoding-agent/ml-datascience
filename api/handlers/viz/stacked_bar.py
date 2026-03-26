"""handle_stacked_bar handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_stacked_bar(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Stacked bar chart — group by one column, color by another."""
    cols = params.get("columns", [])
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    x_col = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else df.columns[0])
    color_col = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)

    if color_col:
        ct = pd.crosstab(df[x_col], df[color_col])
        fig = px.bar(ct, barmode="stack")
    else:
        vc = df[x_col].value_counts().reset_index()
        vc.columns = [x_col, "count"]
        fig = px.bar(vc, x=x_col, y="count")
        fig.update_traces(marker_color="#FB8C3C")

    _style(fig, title=f"{x_col}" + (f" by {color_col}" if color_col else ""), bargap=0.3)
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Stacked bar: {x_col}" + (f" by {color_col}" if color_col else ""))
