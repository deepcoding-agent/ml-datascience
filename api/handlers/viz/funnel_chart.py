"""handle_funnel_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_funnel_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Funnel chart — show stages/flow."""
    col = params.get("column")
    cats = df.select_dtypes(include=["object", "category"]).columns
    col = col if col and col in df.columns else (cats[0] if len(cats) > 0 else df.columns[0])
    vc = df[col].value_counts().head(10).reset_index()
    vc.columns = [col, "count"]
    fig = px.funnel(vc, x="count", y=col)
    _style(fig, title=f"{col} — Funnel")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Funnel chart of '{col}'")
