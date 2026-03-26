"""handle_sensitivity_analysis handler."""
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


def handle_sensitivity_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    target = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    target = target if target and target in num_cols else (num_cols[-1] if num_cols else None)
    if target is None or len(num_cols) < 2:
        return HandlerResult(success=False, error="Need target + features")
    feats = [c for c in num_cols if c != target][:10]
    rows = []
    baseline = float(df[target].mean())
    for f in feats:
        std = float(df[f].std())
        if std == 0: continue
        high = df.copy(); high[f] = high[f] + std
        low = df.copy(); low[f] = low[f] - std
        corr = abs(float(df[f].corr(df[target])))
        rows.append({"feature": f, "baseline": round(baseline, 2), "correlation": round(corr, 4),
                      "sensitivity": round(corr * std, 4), "std": round(std, 4)})
    rows.sort(key=lambda x: x["sensitivity"], reverse=True)
    result_df = pd.DataFrame(rows)
    fig = px.bar(result_df, x="sensitivity", y="feature", orientation="h", text="sensitivity")
    fig.update_traces(marker_color="#FB8C3C", texttemplate="%{text:.4f}", textposition="outside")
    _style(fig, title=f"Sensitivity Analysis — {target}")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"Sensitivity of '{target}' to {len(rows)} features")
