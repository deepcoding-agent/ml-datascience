"""handle_text_normalize handler."""
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


def handle_text_normalize(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Normalize text: strip accents, basic stemming (suffix removal),
    lowercase. Useful for reducing vocabulary before vectorization."""
    col = params.get("column")
    stem = params.get("stem", True)
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    def normalize_text(text: str) -> str:
        # Strip accents
        nfkd = unicodedata.normalize("NFKD", text)
        stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
        # Lowercase
        stripped = stripped.lower()
        # Basic stemming
        if stem:
            words = stripped.split()
            words = [_basic_stem(w) for w in words]
            return " ".join(words)
        return stripped

    for c in text_cols:
        result[c] = result[c].fillna("").astype(str).apply(normalize_text)

    steps = ["accent_strip", "lowercase"]
    if stem:
        steps.append("basic_stemming")

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Normalized {len(text_cols)} column(s): {', '.join(steps)}",
    )
