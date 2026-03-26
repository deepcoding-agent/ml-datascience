"""handle_box_comparison handler."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger
log = get_logger(__name__)

def handle_box_comparison(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Side-by-side box plots comparing numeric columns or groups."""
    col = params.get("column")
    group = params.get("group")
    nums = BaseHandler.get_numeric_cols(df)
    cats = BaseHandler.get_categorical_cols(df)
    if col and group:
        fig = px.box(df, x=group, y=col, color=group)
        _style(fig, title=f"{col} by {group}")
    elif col:
        g = cats[0] if cats else None
        if g:
            fig = px.box(df, x=g, y=col, color=g)
            _style(fig, title=f"{col} by {g}")
        else:
            fig = px.box(df, y=col)
            _style(fig, title=f"Distribution of {col}")
    else:
        if len(nums) < 2:
            return HandlerResult(success=False, error="Need numeric columns to compare")
        melted = df[nums[:8]].melt(var_name="column", value_name="value")
        fig = px.box(melted, x="column", y="value", color="column")
        _style(fig, title="Numeric Columns Comparison")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()], output_type="query",
                         summary="Box comparison chart created")
