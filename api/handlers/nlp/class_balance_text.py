"""handle_class_balance_text handler."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.handlers.nlp._helpers import (
    ENGLISH_STOPWORDS, _POSITIVE_WORDS, _NEGATIVE_WORDS,
    _PATTERNS, _LANG_RANGES, _basic_stem, _get_text_cols,
)
from api.logger import get_logger

log = get_logger(__name__)


def handle_class_balance_text(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Analyze text label/category class balance with visualization.
    Shows class distribution, imbalance ratio, and avg text length per class."""
    col = params.get("column")
    text_col = params.get("text_column")
    if not col or col not in df.columns:
        # Auto-detect: find low-cardinality string column
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        candidates = [c for c in cat_cols if df[c].nunique() <= 50]
        if not candidates:
            return HandlerResult(success=False, error="No categorical/label column found. Specify column= param.")
        col = candidates[0]

    counts = df[col].value_counts()
    imbalance_ratio = round(counts.max() / max(counts.min(), 1), 2)

    rows: list[dict] = []
    for label, count in counts.items():
        row = {"label": str(label), "count": count, "percentage": round(count / len(df) * 100, 2)}
        # If text column provided, compute avg length per class
        if text_col and text_col in df.columns:
            subset = df[df[col] == label][text_col].fillna("").astype(str)
            row["avg_text_length"] = round(subset.str.len().mean(), 1)
            row["avg_word_count"] = round(subset.str.split().str.len().mean(), 1)
        rows.append(row)

    result_df = pd.DataFrame(rows)

    fig = px.bar(
        result_df, x="label", y="count", color="label", text="count",
    )
    fig.update_traces(textposition="outside")
    _style(fig, title=f"Class Balance — {col} (imbalance ratio: {imbalance_ratio}x, n={len(df)})")
    fig.update_layout(xaxis_title="Class", yaxis_title="Count", showlegend=False)

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Class balance for '{col}': {len(counts)} classes, imbalance ratio={imbalance_ratio}x, majority='{counts.index[0]}' ({counts.iloc[0]})",
    )
