"""handle_icicle_chart handler."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger
log = get_logger(__name__)

def handle_icicle_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Icicle chart for hierarchical data (columns as levels)."""
    columns = params.get("columns", [])
    cats = BaseHandler.get_categorical_cols(df)
    if not columns:
        columns = cats[:3] if len(cats) >= 2 else []
    if len(columns) < 2:
        return HandlerResult(success=False, error="Need at least 2 categorical columns for hierarchy")
    agg = df.groupby(columns).size().reset_index(name="count")
    fig = px.icicle(agg, path=columns, values="count")
    _style(fig, title="Icicle Chart")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()], output_type="query",
                         summary=f"Icicle chart with hierarchy: {' → '.join(columns)}")
