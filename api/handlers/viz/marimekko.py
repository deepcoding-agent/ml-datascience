"""handle_marimekko handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_marimekko(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Marimekko/mosaic chart — variable-width stacked bars."""
    cols = params.get("columns", [])
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    x_col = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else None)
    color_col = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)
    if not x_col or not color_col:
        return HandlerResult(success=False, error="Need 2 categorical columns for Marimekko")
    ct = pd.crosstab(df[x_col], df[color_col])
    widths = ct.sum(axis=1)
    widths_norm = widths / widths.sum()
    ct_pct = ct.div(ct.sum(axis=1), axis=0)
    colors = px.colors.qualitative.Set2
    fig = go.Figure()
    x_pos = 0.0
    for i, cat in enumerate(ct.index):
        w = float(widths_norm[cat])
        bottom = 0.0
        for j, sub in enumerate(ct.columns):
            h = float(ct_pct.loc[cat, sub])
            fig.add_trace(go.Bar(x=[x_pos + w / 2], y=[h], width=[w],
                                 base=bottom, name=str(sub),
                                 marker_color=colors[j % len(colors)],
                                 showlegend=(i == 0),
                                 hovertext=f"{cat} / {sub}: {ct.loc[cat, sub]}"))
            bottom += h
        x_pos += w
    fig.update_layout(barmode="stack", xaxis=dict(title=x_col, showticklabels=False),
                      yaxis=dict(title="Proportion", tickformat=".0%"))
    _style(fig, title=f"{x_col} x {color_col} — Marimekko")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Marimekko: {x_col} x {color_col}")
