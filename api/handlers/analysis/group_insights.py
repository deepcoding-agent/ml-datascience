"""handle_group_insights handler."""
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


def handle_group_insights(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compare numeric statistics across groups of a categorical column.
    Shows mean/median/std per group + chart + identifies interesting differences."""
    group_col = params.get("column")
    value_col = params.get("value_column")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not group_col or group_col not in cat_cols:
        group_col = cat_cols[0] if cat_cols else None
    if group_col is None:
        return HandlerResult(success=False, error="No categorical column for grouping")

    if not value_col or value_col not in num_cols:
        value_col = num_cols[0] if num_cols else None
    if value_col is None:
        return HandlerResult(success=False, error="No numeric column for comparison")

    # Limit groups to top 15
    top_groups = df[group_col].value_counts().head(15).index
    subset = df[df[group_col].isin(top_groups)]

    stats = subset.groupby(group_col)[value_col].agg(["count", "mean", "median", "std", "min", "max"])
    stats = stats.round(2).reset_index()
    stats.columns = [group_col, "count", "mean", "median", "std", "min", "max"]
    stats = stats.sort_values("mean", ascending=False)

    # Find most interesting difference
    if len(stats) >= 2:
        best = stats.iloc[0]
        worst = stats.iloc[-1]
        ratio = best["mean"] / worst["mean"] if worst["mean"] != 0 else float("inf")
    else:
        ratio = 1.0

    # Chart: box plot by group
    fig = px.box(subset, x=group_col, y=value_col, color=group_col)
    _style(fig, title=f"{value_col} by {group_col} — Group Comparison (n={len(subset):,})")
    fig.update_layout(showlegend=False, xaxis_title=group_col, yaxis_title=value_col)

    summary = f"Compared {value_col} across {len(stats)} groups of {group_col}. "
    if ratio > 1.5 and len(stats) >= 2:
        summary += f"'{best[group_col]}' has {ratio:.1f}x higher avg than '{worst[group_col]}'. "
    summary += f"Overall range: {stats['mean'].min():,.2f} to {stats['mean'].max():,.2f} (mean)."

    return HandlerResult(
        success=True, result_df=stats, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=summary,
    )
