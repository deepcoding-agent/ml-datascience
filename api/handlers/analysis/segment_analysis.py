"""handle_segment_analysis handler."""
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


def handle_segment_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Auto-segment data by a numeric column into quantile-based groups
    and describe each segment with average features."""
    col = params.get("column")
    n_segments = int(params.get("n", 4))
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not col or col not in num_cols:
        col = num_cols[0] if num_cols else None
    if col is None:
        return HandlerResult(success=False, error="No numeric column for segmentation")

    temp = df.copy()
    try:
        temp["_segment"] = pd.qcut(temp[col], q=n_segments, labels=[f"Q{i+1}" for i in range(n_segments)], duplicates="drop")
    except ValueError:
        temp["_segment"] = pd.cut(temp[col], bins=n_segments, labels=[f"Bin{i+1}" for i in range(n_segments)])

    # Stats per segment
    agg_cols = [c for c in num_cols if c != col][:6]
    agg_dict = {col: ["count", "mean", "min", "max"]}
    for c in agg_cols:
        agg_dict[c] = ["mean"]

    seg_stats = temp.groupby("_segment").agg(agg_dict).round(2)
    seg_stats.columns = ["_".join(c).strip("_") for c in seg_stats.columns]
    seg_stats = seg_stats.reset_index()

    # Chart
    fig = px.bar(
        seg_stats, x="_segment", y=f"{col}_count", text=f"{col}_count",
        color=f"{col}_mean", color_continuous_scale="YlOrRd",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    _style(fig, title=f"Segment Analysis — {col} ({n_segments} segments)")
    fig.update_layout(xaxis_title="Segment", yaxis_title="Count")

    return HandlerResult(
        success=True, result_df=seg_stats, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Segmented {len(df):,} rows into {n_segments} groups by '{col}'",
    )
