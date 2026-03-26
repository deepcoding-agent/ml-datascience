"""handle_deep_profile handler."""
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


def handle_deep_profile(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Deep statistical profile of a single column: distribution, outliers,
    patterns, missing values, quartiles, and auto-visualization."""
    col = params.get("column")
    if not col or col not in df.columns:
        num = df.select_dtypes(include="number").columns.tolist()
        col = num[0] if num else df.columns[0]

    s = df[col].dropna()
    total = len(df[col])
    nulls = df[col].isnull().sum()

    profile: dict = {
        "column": col,
        "dtype": str(df[col].dtype),
        "total_rows": total,
        "non_null": len(s),
        "null_count": nulls,
        "null_pct": round(nulls / max(total, 1) * 100, 2),
        "unique_values": int(df[col].nunique()),
        "unique_pct": round(df[col].nunique() / max(total, 1) * 100, 2),
    }

    charts: list[str] = []

    if pd.api.types.is_numeric_dtype(df[col]):
        profile.update({
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurt()), 4),
            "zeros": int((s == 0).sum()),
            "negatives": int((s < 0).sum()),
        })
        iqr = profile["p75"] - profile["p25"]
        lower = profile["p25"] - 1.5 * iqr
        upper = profile["p75"] + 1.5 * iqr
        outliers = s[(s < lower) | (s > upper)]
        profile["outlier_count"] = len(outliers)
        profile["outlier_pct"] = round(len(outliers) / max(len(s), 1) * 100, 2)
        profile["iqr"] = round(iqr, 4)

        # Distribution chart with box
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                            shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Histogram(x=s, nbinsx=30, marker_color="#FB8C3C", name="Distribution"), row=1, col=1)
        fig.add_trace(go.Box(x=s, marker_color="#2EC4B6", name="Box plot"), row=2, col=1)
        _style(fig, title=f"Deep Profile — {col} (n={len(s):,}, mean={profile['mean']:,.2f})")
        fig.update_layout(showlegend=False, height=450)
        charts.append(fig.to_json())
    else:
        top_values = s.value_counts().head(10)
        profile["top_10_values"] = {str(k): int(v) for k, v in top_values.items()}
        profile["most_common"] = str(top_values.index[0]) if len(top_values) > 0 else "N/A"
        profile["most_common_count"] = int(top_values.iloc[0]) if len(top_values) > 0 else 0

        fig = px.bar(x=top_values.index.astype(str), y=top_values.values, text=top_values.values)
        fig.update_traces(marker_color="#FB8C3C", textposition="outside")
        _style(fig, title=f"Deep Profile — {col} ({profile['unique_values']} unique values)")
        fig.update_layout(xaxis_title=col, yaxis_title="Count")
        charts.append(fig.to_json())

    result_df = pd.DataFrame([profile])

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=charts,
        summary=f"Deep profile of '{col}': {profile.get('dtype')}, {profile.get('non_null')} values, "
                f"{profile.get('null_pct')}% null, {profile.get('unique_values')} unique",
    )
