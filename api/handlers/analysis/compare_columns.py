"""handle_compare_columns handler."""
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


def handle_compare_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Side-by-side comparison of two columns: stats, distribution overlap, correlation."""
    columns = params.get("columns", [])
    if len(columns) < 2:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        columns = num_cols[:2] if len(num_cols) >= 2 else []
    if len(columns) < 2:
        return HandlerResult(success=False, error="Need 2 columns to compare")

    c1, c2 = columns[0], columns[1]
    if c1 not in df.columns or c2 not in df.columns:
        return HandlerResult(success=False, error=f"Column(s) not found: {c1}, {c2}")

    s1, s2 = df[c1].dropna(), df[c2].dropna()

    comp: dict = {"metric": [], c1: [], c2: []}
    if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
        for metric, fn in [("count", "count"), ("mean", "mean"), ("std", "std"),
                           ("min", "min"), ("25%", lambda x: x.quantile(0.25)),
                           ("median", "median"), ("75%", lambda x: x.quantile(0.75)),
                           ("max", "max"), ("skew", "skew")]:
            comp["metric"].append(metric)
            v1 = getattr(s1, fn)() if isinstance(fn, str) else fn(s1)
            v2 = getattr(s2, fn)() if isinstance(fn, str) else fn(s2)
            comp[c1].append(round(float(v1), 4))
            comp[c2].append(round(float(v2), 4))

        corr_val = df[[c1, c2]].corr().iloc[0, 1]
        comp["metric"].append("correlation")
        comp[c1].append(round(float(corr_val), 4))
        comp[c2].append(round(float(corr_val), 4))

        # Overlapping histograms
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=s1, name=c1, opacity=0.6, marker_color="#FB8C3C"))
        fig.add_trace(go.Histogram(x=s2, name=c2, opacity=0.6, marker_color="#2EC4B6"))
        fig.update_layout(barmode="overlay")
        _style(fig, title=f"{c1} vs {c2} — Distribution Comparison")
        charts = [fig.to_json()]
    else:
        for metric in ["count", "unique", "most_common"]:
            comp["metric"].append(metric)
            if metric == "count":
                comp[c1].append(len(s1))
                comp[c2].append(len(s2))
            elif metric == "unique":
                comp[c1].append(int(s1.nunique()))
                comp[c2].append(int(s2.nunique()))
            else:
                comp[c1].append(str(s1.mode().iloc[0]) if len(s1) > 0 else "N/A")
                comp[c2].append(str(s2.mode().iloc[0]) if len(s2) > 0 else "N/A")
        charts = []

    result_df = pd.DataFrame(comp)
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=charts,
        summary=f"Compared '{c1}' vs '{c2}' across {len(result_df)} metrics",
    )
