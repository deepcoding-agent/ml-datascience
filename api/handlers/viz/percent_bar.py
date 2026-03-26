"""handle_percent_bar handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_percent_bar(df: pd.DataFrame, params: dict) -> HandlerResult:
    """100% stacked bar chart — show proportions within each group."""
    cols = params.get("columns", [])
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    x_col = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else df.columns[0])
    color_col = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)
    if color_col:
        ct = pd.crosstab(df[x_col], df[color_col], normalize="index") * 100
        fig = px.bar(ct, barmode="stack")
        fig.update_layout(yaxis_ticksuffix="%")
    else:
        vc = df[x_col].value_counts(normalize=True).head(15).reset_index()
        vc.columns = [x_col, "pct"]
        vc["pct"] = (vc["pct"] * 100).round(1)
        fig = px.bar(vc, x=x_col, y="pct", text="pct")
        fig.update_traces(texttemplate="%{text:.1f}%", marker_color="#FB8C3C")
    _style(fig, title=f"{x_col} — 100% Stacked", bargap=0.3,
           yaxis_title="Percentage (%)")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Percent bar: {x_col}")
