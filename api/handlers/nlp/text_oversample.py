"""handle_text_oversample handler."""
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


def handle_text_oversample(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Oversample minority text classes to balance the dataset.
    Duplicates rows from minority classes to match the majority class count."""
    label_col = params.get("column")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not label_col or label_col not in df.columns:
        candidates = [c for c in cat_cols if df[c].nunique() <= 20]
        label_col = candidates[0] if candidates else None
    if label_col is None:
        return HandlerResult(success=False, error="No label column found for oversampling")

    counts = df[label_col].value_counts()
    max_count = counts.max()
    parts: list[pd.DataFrame] = []
    for label, count in counts.items():
        subset = df[df[label_col] == label]
        if count < max_count:
            repeat = max_count // count
            remainder = max_count % count
            oversampled = pd.concat([subset] * repeat + [subset.head(remainder)], ignore_index=True)
            parts.append(oversampled)
        else:
            parts.append(subset)

    result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Oversampled '{label_col}': {len(df)} → {len(result)} rows (all classes now ≈{max_count})",
    )
