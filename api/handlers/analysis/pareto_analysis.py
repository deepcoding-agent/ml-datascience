"""handle_pareto_analysis handler."""
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


def handle_pareto_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Pareto 80/20 rule analysis on a column."""
    col = params.get("column")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not col or col not in df.columns:
        col = cat_cols[0] if cat_cols else (num_cols[0] if num_cols else None)
    if col is None:
        return HandlerResult(success=False, error="No column found for Pareto analysis")

    counts = df[col].value_counts()
    total = counts.sum()
    pareto = pd.DataFrame({"category": counts.index.astype(str), "count": counts.values})
    pareto["pct"] = round(pareto["count"] / total * 100, 2)
    pareto["cumulative_pct"] = round(pareto["pct"].cumsum(), 2)

    cutoff_idx = int((pareto["cumulative_pct"] >= 80).idxmax())
    n_for_80 = cutoff_idx + 1
    pct_categories = round(n_for_80 / len(pareto) * 100, 1)

    top = pareto.head(20)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=top["category"], y=top["count"], name="Count", marker_color="#FB8C3C"), secondary_y=False)
    fig.add_trace(go.Scatter(x=top["category"], y=top["cumulative_pct"], name="Cumulative %",
                             line=dict(color="#E71D36", width=2), mode="lines+markers"), secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color="#86868B", secondary_y=True, annotation_text="80%")
    _style(fig, title=f"Pareto Analysis — {col}")
    fig.update_layout(yaxis_title="Count", yaxis2_title="Cumulative %")

    return HandlerResult(
        success=True, result_df=pareto, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Pareto: top {n_for_80} of {len(pareto)} categories ({pct_categories:.0f}%) account for 80% of values in '{col}'",
    )
