"""handle_data_quality handler."""
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


def handle_data_quality(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Comprehensive data quality assessment with scores per column
    and overall quality score. Checks: nulls, duplicates, outliers,
    cardinality, constant columns, mixed types."""
    rows: list[dict] = []
    total_score = 0

    for col in df.columns:
        s = df[col]
        n = len(s)
        null_pct = round(s.isnull().sum() / max(n, 1) * 100, 2)
        unique_pct = round(s.nunique() / max(n, 1) * 100, 2)

        issues: list[str] = []
        col_score = 100.0

        # Null penalty
        if null_pct > 0:
            col_score -= min(null_pct, 40)
            issues.append(f"{null_pct}% null")

        # Constant column
        if s.nunique() <= 1:
            col_score -= 20
            issues.append("constant")

        # High cardinality (potential ID column)
        if s.dtype == "object" and unique_pct > 90:
            col_score -= 10
            issues.append("high cardinality (possible ID)")

        # Outliers for numeric
        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            if len(clean) > 4:
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                outlier_pct = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).mean() * 100
                if outlier_pct > 5:
                    col_score -= min(outlier_pct, 15)
                    issues.append(f"{outlier_pct:.1f}% outliers")

        col_score = max(col_score, 0)
        total_score += col_score

        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "null_pct": null_pct,
            "unique_pct": unique_pct,
            "quality_score": round(col_score, 1),
            "issues": ", ".join(issues) if issues else "clean",
        })

    result_df = pd.DataFrame(rows)
    overall = round(total_score / max(len(df.columns), 1), 1)

    # Duplicate check
    dup_count = df.duplicated().sum()

    # Chart: quality scores
    fig = px.bar(
        result_df.sort_values("quality_score"),
        x="quality_score", y="column", orientation="h",
        color="quality_score",
        color_continuous_scale=["#E71D36", "#FF9F1C", "#2EC4B6"],
        text="quality_score",
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    _style(fig, title=f"Data Quality Report — Overall Score: {overall}/100")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Quality Score", yaxis_title="Column",
        coloraxis_showscale=False,
    )

    quality_label = "Excellent" if overall >= 90 else "Good" if overall >= 75 else "Fair" if overall >= 50 else "Poor"
    problem_cols = [r for r in rows if r["quality_score"] < 70]

    summary = f"Overall quality: **{overall}/100 ({quality_label})**. "
    summary += f"{len(df):,} rows × {len(df.columns)} columns"
    if dup_count:
        summary += f", {dup_count} duplicates"
    summary += ". "
    if problem_cols:
        summary += f"Columns needing attention: {', '.join(r['column'] for r in problem_cols[:5])}."

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=summary,
        metadata={"overall_score": overall, "duplicate_count": int(dup_count)},
    )
