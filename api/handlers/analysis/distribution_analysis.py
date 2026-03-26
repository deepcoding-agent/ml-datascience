"""handle_distribution_analysis handler."""
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


def handle_distribution_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Analyze distribution shape (skew, kurtosis, normality) per numeric column."""
    from scipy import stats as sp_stats

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns found")

    rows: list[dict] = []
    for c in num_cols:
        s = df[c].dropna()
        if len(s) < 8:
            continue
        sk = float(s.skew())
        ku = float(s.kurt())
        try:
            _, p_shapiro = sp_stats.shapiro(s.sample(min(len(s), 5000), random_state=42))
        except Exception:
            p_shapiro = 0.0

        shape = "Normal" if abs(sk) < 0.5 and abs(ku) < 1 else "Skewed" if abs(sk) > 1 else "Heavy-tailed" if ku > 3 else "Moderate"
        rows.append({
            "column": c, "skewness": round(sk, 3), "kurtosis": round(ku, 3),
            "shapiro_p": round(float(p_shapiro), 4), "normal": p_shapiro > 0.05, "shape": shape,
        })

    if not rows:
        return HandlerResult(success=False, error="No columns with enough data for distribution analysis")

    result_df = pd.DataFrame(rows)
    non_normal = result_df[~result_df["normal"]]

    fig = px.bar(result_df, x="column", y="skewness", color="shape", text="skewness")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    _style(fig, title=f"Distribution Shape — {len(result_df)} numeric columns")

    most_skewed = result_df.loc[result_df["skewness"].abs().idxmax(), "column"]
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"{len(non_normal)}/{len(result_df)} columns are non-normal (Shapiro p<0.05). Most skewed: {most_skewed}",
    )
