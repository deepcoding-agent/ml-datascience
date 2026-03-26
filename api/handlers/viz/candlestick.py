"""handle_candlestick handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style, _group_pie, _THEME
from api.logger import get_logger

log = get_logger(__name__)


def handle_candlestick(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Candlestick / OHLC chart — needs open/high/low/close columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    o = params.get("open") or next((c for c in num_cols if "open" in c.lower()), None)
    h = params.get("high") or next((c for c in num_cols if "high" in c.lower()), None)
    lo = params.get("low") or next((c for c in num_cols if "low" in c.lower()), None)
    c = params.get("close") or next((c for c in num_cols if "close" in c.lower()), None)
    if not all([o, h, lo, c]):
        if len(num_cols) >= 4:
            o, h, lo, c = num_cols[:4]
        else:
            return HandlerResult(success=False, error="Need open/high/low/close columns for candlestick")
    dt_cols = df.select_dtypes(include="datetime").columns.tolist()
    x = dt_cols[0] if dt_cols else df.index
    fig = go.Figure(go.Candlestick(x=x if isinstance(x, pd.Index) else df[x],
                                   open=df[o], high=df[h], low=df[lo], close=df[c]))
    _style(fig, title="Candlestick Chart")
    return HandlerResult(success=True, charts_plotly=[fig.to_json()],
                         summary="Candlestick chart")
