"""handle_imbalance_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_imbalance_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Class distribution analysis with imbalance ratio and resampling recommendation."""
    target_col = params.get("column") or params.get("target")

    if not target_col or target_col not in df.columns:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_low = [c for c in df.select_dtypes(include="number").columns if df[c].nunique() <= 10]
        target_col = (num_low + cat_cols + [df.columns[-1]])[0]

    y = df[target_col].dropna()
    vc = y.value_counts()
    n_classes = len(vc)
    majority = int(vc.max())
    minority = int(vc.min())
    ratio = round(majority / max(minority, 1), 2)

    if ratio < 1.5:
        severity = "Balanced"
        recommendation = "No resampling needed. Proceed with standard training."
    elif ratio < 3:
        severity = "Mild imbalance"
        recommendation = "Consider class_weight='balanced' in your model, or try random oversampling."
    elif ratio < 10:
        severity = "Moderate imbalance"
        recommendation = "Use SMOTE oversampling or random undersampling. Set class_weight='balanced'."
    else:
        severity = "Severe imbalance"
        recommendation = "Use SMOTE or ADASYN. Consider combining oversampling minority + undersampling majority. Always use stratified cross-validation."

    result_df = pd.DataFrame({
        "class": vc.index.astype(str),
        "count": vc.values,
        "percentage": (vc.values / len(y) * 100).round(2),
    })

    # Chart: bar + pie side by side
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]],
                        subplot_titles=["Class Counts", "Class Distribution"])
    colors = px.colors.qualitative.Set2[:n_classes]
    fig.add_trace(go.Bar(x=result_df["class"], y=result_df["count"],
                         marker_color=colors, text=result_df["percentage"].astype(str) + "%",
                         textposition="outside", name="Count"), row=1, col=1)
    fig.add_trace(go.Pie(labels=result_df["class"], values=result_df["count"],
                         marker_colors=colors, textinfo="label+percent", name="Dist"), row=1, col=2)
    _style(fig, title=f"Imbalance Report: {target_col} — {severity} (ratio {ratio}:1)")

    summary = (
        f"Imbalance report for '{target_col}': {n_classes} classes, "
        f"ratio {ratio}:1 (majority {majority}, minority {minority}). "
        f"Severity: {severity}. Recommendation: {recommendation}"
    )
    return HandlerResult(
        success=True, result_df=result_df, charts_plotly=[fig.to_json()],
        summary=summary,
        metadata={"ratio": ratio, "severity": severity, "recommendation": recommendation},
    )
