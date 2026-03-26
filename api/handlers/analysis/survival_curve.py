"""handle_survival_curve handler."""
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


def handle_survival_curve(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in num_cols else (num_cols[0] if num_cols else None)
    if col is None:
        return HandlerResult(success=False, error="No numeric column found")
    s = df[col].dropna().sort_values().reset_index(drop=True)
    n = len(s)
    survival = [(1 - (i + 1) / n) for i in range(n)]
    result_df = pd.DataFrame({col: s.values, "survival_prob": [round(x, 4) for x in survival]})
    fig = go.Figure(go.Scatter(x=s.values, y=survival, mode="lines", line=dict(color="#FB8C3C", width=2)))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#86868B")
    _style(fig, title=f"Survival Curve — {col}")
    fig.update_layout(xaxis_title=col, yaxis_title="Survival Probability")
    median_val = float(s.median())
    return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"Survival curve for '{col}': median={median_val:.2f}, range [{s.min():.2f}, {s.max():.2f}]")
