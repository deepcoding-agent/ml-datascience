"""handle_feature_selection_auto handler."""
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


def handle_feature_selection_auto(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Auto feature selection: variance filter + correlation filter + mutual info."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder

    target_col = params.get("column") or params.get("target")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")

    if not target_col or target_col not in df.columns:
        target_col = num_cols[-1]

    feature_cols = [c for c in num_cols if c != target_col]
    clean = df[feature_cols + [target_col]].dropna()
    if len(clean) < 10:
        return HandlerResult(success=False, error="Need at least 10 non-null rows")

    X = clean[feature_cols]
    y = clean[target_col]

    rows: list[dict] = []
    for c in feature_cols:
        var = float(X[c].var())
        corr_val = float(X[c].corr(y)) if pd.api.types.is_numeric_dtype(y) else 0.0
        rows.append({"feature": c, "variance": round(var, 4), "abs_corr_target": round(abs(corr_val), 4)})

    try:
        if y.nunique() <= 20:
            le = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            mi = mutual_info_classif(X, y_enc, random_state=42)
        else:
            mi = mutual_info_regression(X, y, random_state=42)
        for i, c in enumerate(feature_cols):
            rows[i]["mutual_info"] = round(float(mi[i]), 4)
    except Exception:
        for r in rows:
            r["mutual_info"] = 0.0

    result_df = pd.DataFrame(rows).sort_values("mutual_info", ascending=False)
    result_df["rank"] = range(1, len(result_df) + 1)

    fig = px.bar(
        result_df.head(15), x="mutual_info", y="feature", orientation="h",
        text="mutual_info", color="abs_corr_target", color_continuous_scale="YlOrRd",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    _style(fig, title=f"Feature Importance (target={target_col})")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Ranked {len(feature_cols)} features by importance for '{target_col}'. Top: {result_df.iloc[0]['feature']}",
    )
