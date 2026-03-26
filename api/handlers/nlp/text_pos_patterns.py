"""handle_text_pos_patterns handler."""
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


def handle_text_pos_patterns(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Detect POS-like surface patterns: all_caps ratio, capitalized ratio,
    numeric word ratio per row. Lightweight alternative to full POS tagging."""
    col = params.get("column")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    result = df.copy()
    created: list[str] = []
    for c in text_cols:
        s = result[c].fillna("").astype(str)

        def _patterns(text: str) -> dict:
            words = re.findall(r"\b\w+\b", text)
            n = max(len(words), 1)
            all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
            capitalized = sum(1 for w in words if w and w[0].isupper() and not w.isupper())
            numeric = sum(1 for w in words if re.match(r"^-?\d+\.?\d*$", w))
            return {
                "all_caps_ratio": round(all_caps / n, 4),
                "capitalized_ratio": round(capitalized / n, 4),
                "numeric_word_ratio": round(numeric / n, 4),
            }

        stats = s.apply(_patterns).apply(pd.Series)
        for feat in ["all_caps_ratio", "capitalized_ratio", "numeric_word_ratio"]:
            result[f"{c}_{feat}"] = stats[feat]
            created.append(f"{c}_{feat}")

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Extracted {len(created)} POS-pattern features from {len(text_cols)} column(s)",
    )
