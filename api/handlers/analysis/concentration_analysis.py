"""handle_concentration_analysis handler."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_concentration_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in num_cols else (num_cols[0] if num_cols else None)
    if col is None:
        return HandlerResult(success=False, error="No numeric column found")
    s = df[col].dropna().sort_values().values
    n = len(s)
    cum = np.cumsum(s) / s.sum()
    pct = np.arange(1, n + 1) / n
    gini = float(1 - 2 * np.trapz(cum, pct))
    top10_share = float(s[int(n * 0.9):].sum() / s.sum() * 100)
    top20_share = float(s[int(n * 0.8):].sum() / s.sum() * 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pct * 100, y=cum * 100, mode="lines", name="Lorenz", line=dict(color="#FB8C3C", width=2)))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Equality", line=dict(color="#86868B", dash="dash")))
    _style(fig, title=f"Lorenz Curve — {col} (Gini={gini:.3f})")
    fig.update_layout(xaxis_title="Population %", yaxis_title="Cumulative %")
    result_df = pd.DataFrame({"metric": ["gini", "top_10%_share", "top_20%_share"], "value": [round(gini, 4), round(top10_share, 2), round(top20_share, 2)]})
    return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"Concentration of '{col}': Gini={gini:.3f}, top 20% holds {top20_share:.1f}%")
