"""handle_class_balance handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_class_balance(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Analyze target class distribution — important before classification."""
    col = params.get("column")
    if not col or col not in df.columns:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        # Prefer low-cardinality numeric (likely a label) or first categorical
        for c in num_cols:
            if df[c].nunique() <= 10:
                col = c
                break
        if not col:
            col = cat_cols[0] if cat_cols else df.columns[-1]

    vc = df[col].value_counts()
    result = pd.DataFrame({
        "class": vc.index.astype(str),
        "count": vc.values,
        "percentage": (vc.values / len(df) * 100).round(2),
    })

    ratio = vc.max() / vc.min() if vc.min() > 0 else float("inf")
    balanced = "Balanced" if ratio < 2 else ("Moderate imbalance" if ratio < 5 else "Severe imbalance")

    fig = px.bar(result, x="class", y="count", text="percentage")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                      marker_color="#FB8C3C")
    _style(fig, title=f"Class Balance: {col}", bargap=0.3)

    return HandlerResult(
        success=True, result_df=result, charts_plotly=[fig.to_json()],
        summary=f"Class balance '{col}': {len(vc)} classes, ratio {ratio:.1f}:1 — {balanced}",
    )
