"""handle_sankey_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_sankey_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Sankey flow diagram between two categorical columns."""
    cols = params.get("columns", [])
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    src_col = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else None)
    tgt_col = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)
    if not src_col or not tgt_col:
        return HandlerResult(success=False, error="Need 2 categorical columns for Sankey")
    flow = df.groupby([src_col, tgt_col]).size().reset_index(name="count")
    flow = flow.nlargest(20, "count")
    labels = list(set(flow[src_col].tolist() + flow[tgt_col].tolist()))
    label_idx = {l: i for i, l in enumerate(labels)}
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, color="#FB8C3C", pad=15),
        link=dict(source=[label_idx[s] for s in flow[src_col]],
                  target=[label_idx[t] for t in flow[tgt_col]],
                  value=flow["count"].tolist())))
    _style(fig, title=f"{src_col} → {tgt_col} — Flow")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Sankey: {src_col} → {tgt_col}")
