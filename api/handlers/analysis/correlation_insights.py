"""handle_correlation_insights handler."""
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


def handle_correlation_insights(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Find and explain the most interesting correlations in the dataset.
    Returns top positive/negative correlations with scatter plots."""
    n = int(params.get("n", 10))
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")

    corr = df[num_cols].corr()

    # Extract top correlations (excluding self-correlation)
    pairs: list[tuple[str, str, float]] = []
    for i, c1 in enumerate(num_cols):
        for j, c2 in enumerate(num_cols):
            if i < j:
                r = corr.loc[c1, c2]
                if not np.isnan(r):
                    pairs.append((c1, c2, round(float(r), 4)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pairs = pairs[:n]

    result_df = pd.DataFrame(top_pairs, columns=["column_1", "column_2", "correlation"])
    result_df["strength"] = result_df["correlation"].abs().apply(
        lambda r: "Very Strong" if r >= 0.8 else "Strong" if r >= 0.6
        else "Moderate" if r >= 0.4 else "Weak"
    )
    result_df["direction"] = result_df["correlation"].apply(lambda r: "Positive" if r > 0 else "Negative")

    # Charts: scatter of top 2 pairs
    charts: list[str] = []
    for c1, c2, r in top_pairs[:2]:
        fig = px.scatter(
            df, x=c1, y=c2, trendline="ols",
            opacity=0.5,
        )
        fig.update_traces(marker_color="#FB8C3C")
        _style(fig, title=f"{c1} vs {c2} (r={r:+.3f})")
        fig.update_layout(xaxis_title=c1, yaxis_title=c2)
        charts.append(fig.to_json())

    # Summary
    strong = [p for p in top_pairs if abs(p[2]) >= 0.6]
    summary = f"Analyzed {len(pairs)} column pairs. "
    if strong:
        summary += f"{len(strong)} strong correlations found. "
        top = strong[0]
        direction = "positively" if top[2] > 0 else "negatively"
        summary += f"Strongest: {top[0]} & {top[1]} (r={top[2]:+.3f}, {direction} correlated)."
    else:
        summary += "No strong correlations (|r| ≥ 0.6) found."

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=charts,
        summary=summary,
    )
