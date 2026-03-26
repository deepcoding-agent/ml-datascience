"""handle_auto_eda handler."""
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


def handle_auto_eda(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Automated exploratory data analysis — generates key findings
    about the dataset: shape, quality issues, distributions, correlations,
    and actionable recommendations."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    findings: list[str] = []
    recommendations: list[str] = []

    # 1. Shape
    findings.append(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns ({len(num_cols)} numeric, {len(cat_cols)} categorical)")

    # 2. Missing values
    null_summary = df.isnull().sum()
    null_cols = null_summary[null_summary > 0]
    if len(null_cols) > 0:
        worst = null_cols.idxmax()
        worst_pct = null_cols.max() / len(df) * 100
        findings.append(f"Missing values: {len(null_cols)} columns affected, worst is '{worst}' ({worst_pct:.1f}%)")
        if worst_pct > 50:
            recommendations.append(f"Consider dropping '{worst}' (>50% missing)")
        else:
            recommendations.append(f"Fill missing values in {len(null_cols)} columns")

    # 3. Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        findings.append(f"Duplicates: {dup_count:,} duplicate rows ({dup_count/len(df)*100:.1f}%)")
        recommendations.append("Remove duplicate rows")

    # 4. Correlations
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        high_corr = []
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                if i < j and abs(corr.loc[c1, c2]) >= 0.7:
                    high_corr.append((c1, c2, round(float(corr.loc[c1, c2]), 3)))
        if high_corr:
            high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
            top = high_corr[0]
            findings.append(f"Strong correlations: {len(high_corr)} pairs, strongest: {top[0]} & {top[1]} (r={top[2]:+.3f})")

    # 5. Skewed columns
    if num_cols:
        skewed = []
        for c in num_cols:
            sk = df[c].skew()
            if abs(sk) > 1.5:
                skewed.append((c, round(float(sk), 2)))
        if skewed:
            findings.append(f"Highly skewed: {', '.join(c for c, _ in skewed[:3])} — consider log/power transform")
            recommendations.append("Apply log_transform or power_transform to skewed columns")

    # 6. Constant columns
    constant = [c for c in df.columns if df[c].nunique() <= 1]
    if constant:
        findings.append(f"Constant columns: {', '.join(constant)} — no information, safe to drop")
        recommendations.append(f"Drop constant columns: {', '.join(constant)}")

    # 7. Categorical analysis
    for c in cat_cols[:3]:
        unique = df[c].nunique()
        if unique > 50:
            findings.append(f"'{c}' has high cardinality ({unique} unique) — may need encoding or grouping")
        elif unique == 2:
            findings.append(f"'{c}' is binary — good candidate for label encoding")

    # Build report table
    report_rows = [{"type": "Finding", "detail": f} for f in findings]
    report_rows += [{"type": "Recommendation", "detail": r} for r in recommendations]
    result_df = pd.DataFrame(report_rows)

    # Chart: null heatmap if nulls exist
    charts: list[str] = []
    if len(null_cols) > 0:
        null_pcts = (df.isnull().sum() / len(df) * 100).round(1)
        null_pcts = null_pcts[null_pcts > 0].sort_values(ascending=True)
        fig = px.bar(
            x=null_pcts.values, y=null_pcts.index, orientation="h",
            text=[f"{v:.1f}%" for v in null_pcts.values],
        )
        fig.update_traces(marker_color="#E71D36", textposition="outside")
        _style(fig, title=f"Missing Values — {len(null_cols)} columns affected")
        fig.update_layout(xaxis_title="Null %", yaxis_title="Column")
        charts.append(fig.to_json())

    summary = "\n".join(f"• {f}" for f in findings)
    if recommendations:
        summary += "\n\n**Recommendations:**\n" + "\n".join(f"→ {r}" for r in recommendations)

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=charts,
        summary=summary,
    )
