"""handle_feature_drift handler."""
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


def handle_feature_drift(df: pd.DataFrame, params: dict) -> HandlerResult:
    num_cols = df.select_dtypes(include="number").columns.tolist()[:10]
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns")
    mid = len(df) // 2
    first, second = df.iloc[:mid], df.iloc[mid:]
    rows = []
    for c in num_cols:
        m1, m2 = float(first[c].mean()), float(second[c].mean())
        s1, s2 = float(first[c].std()), float(second[c].std())
        drift_pct = abs(m1 - m2) / max(abs(m1), 1e-10) * 100
        rows.append({"column": c, "first_half_mean": round(m1, 4), "second_half_mean": round(m2, 4),
                      "drift_pct": round(drift_pct, 2), "drifted": drift_pct > 10})
    result_df = pd.DataFrame(rows)
    drifted = sum(1 for r in rows if r["drifted"])
    fig = px.bar(result_df, x="column", y="drift_pct", color="drifted", text="drift_pct",
                 color_discrete_map={True: "#E71D36", False: "#2EC4B6"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    _style(fig, title=f"Feature Drift Analysis — {drifted}/{len(rows)} features drifted")
    return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"Feature drift: {drifted}/{len(rows)} features show >10% mean drift between halves")
