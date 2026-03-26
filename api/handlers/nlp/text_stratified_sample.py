"""handle_text_stratified_sample handler."""
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


def handle_text_stratified_sample(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Take a stratified random sample maintaining label distribution.
    Useful for creating balanced train/test splits or review subsets."""
    label_col = params.get("column")
    n = int(params.get("n", 100))
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not label_col or label_col not in df.columns:
        candidates = [c for c in cat_cols if df[c].nunique() <= 30]
        label_col = candidates[0] if candidates else None
    if label_col is None:
        return HandlerResult(success=False, error="No label column for stratified sampling")

    counts = df[label_col].value_counts()
    n_classes = len(counts)
    per_class = max(1, n // n_classes)

    parts: list[pd.DataFrame] = []
    for label in counts.index:
        subset = df[df[label_col] == label]
        sample_n = min(per_class, len(subset))
        parts.append(subset.sample(n=sample_n, random_state=42))

    result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    new_dist = result[label_col].value_counts().to_dict()

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Stratified sample: {len(df)} → {len(result)} rows, {n_classes} classes. Distribution: {new_dist}",
    )
