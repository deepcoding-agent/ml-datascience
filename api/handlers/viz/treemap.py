"""handle_treemap handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_treemap(df: pd.DataFrame, params: dict) -> HandlerResult:
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col = params.get("column") or (cats[0] if cats else None)
    if not col or col not in df.columns:
        return HandlerResult(success=False, error="No categorical column for treemap")
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, "count"]
    fig = px.treemap(vc, path=[col], values="count")
    _style(fig, title=f"{col}")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary=f"Treemap of '{col}'")
