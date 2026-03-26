"""handle_char_features handler."""
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


def handle_char_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract character-level features: punctuation count/ratio, digit ratio,
    uppercase ratio, special char ratio, whitespace ratio, avg word length."""
    col = params.get("column")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    created: list[str] = []
    for c in text_cols:
        s = result[c].fillna("").astype(str)
        length = s.str.len().replace(0, 1)

        result[f"{c}_char_count"] = s.str.len()
        result[f"{c}_punct_count"] = s.str.count(r"[^\w\s]")
        result[f"{c}_punct_ratio"] = (result[f"{c}_punct_count"] / length).round(4)
        result[f"{c}_digit_ratio"] = (s.str.count(r"\d") / length).round(4)
        result[f"{c}_upper_ratio"] = (s.str.count(r"[A-Z]") / length).round(4)
        result[f"{c}_lower_ratio"] = (s.str.count(r"[a-z]") / length).round(4)
        result[f"{c}_space_ratio"] = (s.str.count(r"\s") / length).round(4)
        result[f"{c}_special_ratio"] = (s.str.count(r"[^a-zA-Z0-9\s]") / length).round(4)
        # Average word length
        words = s.str.findall(r"\b\w+\b")
        result[f"{c}_avg_word_len"] = words.apply(
            lambda ws: round(np.mean([len(w) for w in ws]), 2) if ws else 0.0
        )
        created.extend([
            f"{c}_char_count", f"{c}_punct_count", f"{c}_punct_ratio",
            f"{c}_digit_ratio", f"{c}_upper_ratio", f"{c}_lower_ratio",
            f"{c}_space_ratio", f"{c}_special_ratio", f"{c}_avg_word_len",
        ])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Extracted {len(created)} character-level features from {len(text_cols)} column(s)",
    )
