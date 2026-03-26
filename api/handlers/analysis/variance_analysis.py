"""handle_variance_analysis handler."""
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


def handle_variance_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Variance contribution per feature (% of total variance)."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns found")

    variances: dict[str, float] = {}
    for c in num_cols:
        v = df[c].var(skipna=True)
        if pd.notna(v):
            variances[c] = float(v)

    total_var = sum(variances.values())
    rows = [{"feature": k, "variance": round(v, 4), "pct_of_total": round(v / max(total_var, 1e-10) * 100, 2)}
            for k, v in sorted(variances.items(), key=lambda x: x[1], reverse=True)]

    result_df = pd.DataFrame(rows)

    fig = px.bar(result_df.head(15), x="pct_of_total", y="feature", orientation="h",
                 text="pct_of_total", color="pct_of_total", color_continuous_scale="YlOrRd")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    _style(fig, title=f"Variance Contribution — {len(num_cols)} features")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)

    top3_pct = sum(r["pct_of_total"] for r in rows[:3])
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Top variance contributor: {rows[0]['feature']} ({rows[0]['pct_of_total']:.1f}%). Top 3 account for {top3_pct:.1f}%",
    )
