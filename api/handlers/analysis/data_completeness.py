"""handle_data_completeness handler."""
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


def handle_data_completeness(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Data completeness scorecard per column + overall score."""
    rows: list[dict] = []
    for c in df.columns:
        n_total = len(df)
        n_present = int(df[c].count())
        n_missing = n_total - n_present
        completeness = round(n_present / max(n_total, 1) * 100, 2)
        grade = "A" if completeness >= 95 else "B" if completeness >= 80 else "C" if completeness >= 50 else "F"
        rows.append({
            "column": c, "total": n_total, "present": n_present,
            "missing": n_missing, "completeness_pct": completeness, "grade": grade,
        })

    result_df = pd.DataFrame(rows).sort_values("completeness_pct", ascending=True)
    overall = round(float(result_df["completeness_pct"].mean()), 1)
    overall_grade = "A" if overall >= 95 else "B" if overall >= 80 else "C" if overall >= 50 else "F"

    fig = px.bar(result_df, x="completeness_pct", y="column", orientation="h",
                 color="grade", text="completeness_pct",
                 color_discrete_map={"A": "#2EC4B6", "B": "#FB8C3C", "C": "#FF9F1C", "F": "#E71D36"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    _style(fig, title=f"Data Completeness Scorecard — Overall: {overall}% ({overall_grade})")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_range=[0, 105])

    incomplete = [r for r in rows if r["completeness_pct"] < 100]
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Data completeness: {overall}% ({overall_grade}). {len(incomplete)}/{len(rows)} columns have missing values.",
    )
