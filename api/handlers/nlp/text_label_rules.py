"""handle_text_label_rules handler."""
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


def handle_text_label_rules(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create text labels based on keyword rules.
    mapping: dict of {label: [keyword1, keyword2, ...]}."""
    col = params.get("column")
    mapping = params.get("mapping")  # {"positive": ["good","great"], "negative": ["bad","poor"]}
    default_label = params.get("default", "other")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")
    if not mapping or not isinstance(mapping, dict):
        return HandlerResult(success=False, error="Provide mapping= dict: {label: [keywords]}")

    target = text_cols[0]
    result = df.copy()
    s = result[target].fillna("").astype(str).str.lower()

    def classify(text: str) -> str:
        for label, keywords in mapping.items():
            for kw in keywords:
                if kw.lower() in text:
                    return str(label)
        return default_label

    result[f"{target}_label"] = s.apply(classify)
    dist = result[f"{target}_label"].value_counts().to_dict()

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Labeled {len(result)} rows using {len(mapping)} rules. Distribution: {dist}",
    )
