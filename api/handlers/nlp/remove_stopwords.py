"""handle_remove_stopwords handler."""
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


def handle_remove_stopwords(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Remove English stopwords from text columns."""
    col = params.get("column")
    extra_stops = set(params.get("extra_words", []))
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    stops = ENGLISH_STOPWORDS | extra_stops

    for c in text_cols:
        s = result[c].fillna("").astype(str)
        result[c] = s.apply(
            lambda t: " ".join(w for w in t.split() if w.lower() not in stops)
        )

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Removed stopwords from {len(text_cols)} column(s) ({len(stops)} stopwords)",
    )
