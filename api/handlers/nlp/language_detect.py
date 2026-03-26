"""handle_language_detect handler."""
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


def handle_language_detect(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Detect language per row based on Unicode character ranges.
    Creates {col}_language column."""
    col = params.get("column")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    def detect_lang(text: str) -> str:
        if not text.strip():
            return "empty"
        char_counts: dict[str, int] = {}
        for ch in text:
            cp = ord(ch)
            for lo, hi, lang in _LANG_RANGES:
                if lo <= cp <= hi:
                    char_counts[lang] = char_counts.get(lang, 0) + 1
                    break
        if not char_counts:
            return "unknown"
        return max(char_counts, key=char_counts.get)  # type: ignore[arg-type]

    created: list[str] = []
    for c in text_cols:
        lang_col = f"{c}_language"
        result[lang_col] = result[c].fillna("").astype(str).apply(detect_lang)
        created.append(lang_col)

    # Chart
    charts: list[str] = []
    if created:
        lang_counts = result[created[0]].value_counts()
        fig = px.pie(
            values=lang_counts.values, names=lang_counts.index,
        )
        _style(fig, title=f"Language Distribution — {text_cols[0]} (n={len(result)})")
        charts.append(fig.to_json())

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        charts_plotly=charts,
        summary=f"Detected language in {len(text_cols)} column(s): {', '.join(created)}",
    )
