"""handle_change_point_detect handler."""
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


def handle_change_point_detect(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Detect change points in a numeric series using sliding window mean diff."""
    col = params.get("column")
    window = int(params.get("window", 10))
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not col or col not in num_cols:
        col = num_cols[0] if num_cols else None
    if col is None:
        return HandlerResult(success=False, error="No numeric column found")

    s = df[col].dropna().reset_index(drop=True)
    if len(s) < window * 2:
        return HandlerResult(success=False, error=f"Need at least {window * 2} data points, got {len(s)}")

    diffs: list[float] = []
    for i in range(window, len(s) - window):
        left_mean = float(s.iloc[i - window:i].mean())
        right_mean = float(s.iloc[i:i + window].mean())
        diffs.append(abs(right_mean - left_mean))

    diffs_s = pd.Series(diffs)
    threshold = float(diffs_s.mean() + 2 * diffs_s.std())
    change_points = [i + window for i, d in enumerate(diffs) if d > threshold]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=s.values, mode="lines", name=col, line=dict(color="#86868B", width=1)))
    for cp in change_points[:10]:
        fig.add_vline(x=cp, line_dash="dash", line_color="#E71D36", annotation_text=f"CP@{cp}")
    _style(fig, title=f"Change Point Detection — {col} ({len(change_points)} points)")

    cp_limited = change_points[:20]
    result_df = pd.DataFrame({
        "change_point_index": cp_limited,
        "value": [round(float(s.iloc[cp]), 3) for cp in cp_limited],
    })
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Found {len(change_points)} change points in '{col}' (window={window}, threshold=mean+2*std)",
    )
