"""handle_percentile_analysis handler."""
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


def handle_percentile_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compare stats across percentile bands (Q1-Q4)."""
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not col or col not in num_cols:
        col = num_cols[0] if num_cols else None
    if col is None:
        return HandlerResult(success=False, error="No numeric column found")

    temp = df[[col]].dropna().copy()
    try:
        temp["quartile"] = pd.qcut(temp[col], q=4, labels=["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"], duplicates="drop")
    except ValueError:
        return HandlerResult(success=False, error=f"Cannot create quartiles for '{col}' — too few unique values")

    stats = temp.groupby("quartile")[col].agg(["count", "mean", "min", "max", "std"]).round(3).reset_index()

    fig = px.box(temp, x="quartile", y=col, color="quartile")
    _style(fig, title=f"Percentile Analysis — {col}")
    fig.update_layout(showlegend=False)

    return HandlerResult(
        success=True, result_df=stats, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Percentile breakdown for '{col}': Q1 mean={stats.iloc[0]['mean']:.3f}, Q4 mean={stats.iloc[-1]['mean']:.3f}",
    )
