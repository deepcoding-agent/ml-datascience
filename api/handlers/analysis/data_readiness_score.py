"""handle_data_readiness_score handler."""
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


def handle_data_readiness_score(df: pd.DataFrame, params: dict) -> HandlerResult:
    scores = {}
    # Completeness
    null_pct = df.isnull().mean().mean() * 100
    scores["completeness"] = round(max(0, 100 - null_pct * 2), 1)
    # Duplicates
    dup_pct = df.duplicated().mean() * 100
    scores["uniqueness"] = round(max(0, 100 - dup_pct * 2), 1)
    # Consistency (no constant cols, no mixed types)
    constant_pct = sum(1 for c in df.columns if df[c].nunique() <= 1) / max(len(df.columns), 1) * 100
    scores["consistency"] = round(max(0, 100 - constant_pct * 5), 1)
    # Balance (for categoricals)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        imb = max(df[c].value_counts().max() / max(df[c].value_counts().min(), 1) for c in cat_cols[:3])
        scores["balance"] = round(max(0, 100 - (imb - 1) * 10), 1)
    else:
        scores["balance"] = 100.0
    # Size
    scores["size"] = min(100.0, round(len(df) / 10, 1))  # 1000 rows = 100%
    overall = round(sum(scores.values()) / len(scores), 1)
    grade = "A" if overall >= 90 else "B" if overall >= 75 else "C" if overall >= 50 else "D"
    result_df = pd.DataFrame([{"dimension": k, "score": v} for k, v in scores.items()])
    result_df = pd.concat([result_df, pd.DataFrame([{"dimension": "OVERALL", "score": overall}])], ignore_index=True)
    fig = px.bar(result_df, x="score", y="dimension", orientation="h", text="score",
                 color="score", color_continuous_scale=["#E71D36", "#FF9F1C", "#2EC4B6"])
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    _style(fig, title=f"ML Data Readiness — {overall}/100 ({grade})")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_range=[0, 110], coloraxis_showscale=False)
    return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"Data readiness score: {overall}/100 ({grade}). {', '.join(f'{k}={v}' for k, v in scores.items())}")
