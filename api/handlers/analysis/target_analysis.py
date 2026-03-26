"""handle_target_analysis handler."""
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


def handle_target_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Analyze relationship between each feature and a target column."""
    target = params.get("column") or params.get("target")
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not target or target not in num_cols:
        target = num_cols[-1] if num_cols else None
    if target is None:
        return HandlerResult(success=False, error="No numeric target column found")

    features = [c for c in num_cols if c != target]
    if not features:
        return HandlerResult(success=False, error="No feature columns found besides target")

    rows: list[dict] = []
    for f in features:
        clean = df[[f, target]].dropna()
        if len(clean) < 5:
            continue
        corr_val = float(clean[f].corr(clean[target]))
        rows.append({"feature": f, "correlation": round(corr_val, 4),
                     "abs_corr": round(abs(corr_val), 4),
                     "direction": "Positive" if corr_val > 0 else "Negative"})

    if not rows:
        return HandlerResult(success=False, error=f"No valid features for target '{target}'")

    result_df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)

    fig = px.bar(result_df.head(15), x="correlation", y="feature", orientation="h",
                 color="correlation", color_continuous_scale="RdBu_r", text="correlation")
    fig.update_traces(texttemplate="%{text:+.3f}", textposition="outside")
    _style(fig, title=f"Feature-Target Correlation (target={target})")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    top_feat = result_df.iloc[0]["feature"]
    top_corr = result_df.iloc[0]["correlation"]
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Target '{target}': strongest predictor is '{top_feat}' (r={top_corr:+.3f})",
    )
