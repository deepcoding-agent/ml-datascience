"""handle_text_remove_rare handler."""
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


def handle_text_remove_rare(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Remove words appearing below a frequency threshold from text."""
    col = params.get("column")
    min_freq = int(params.get("min_freq", 2))
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    result = df.copy()
    for c in text_cols:
        s = result[c].fillna("").astype(str).str.lower()
        # Build corpus-wide word frequencies
        word_freq: Counter = Counter()
        for text in s:
            word_freq.update(re.findall(r"\b\w+\b", text))

        rare_words = {w for w, cnt in word_freq.items() if cnt < min_freq}

        def _remove(text: str) -> str:
            words = text.split()
            return " ".join(w for w in words if w.lower() not in rare_words)

        result[c] = result[c].fillna("").astype(str).apply(_remove)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Removed words with freq < {min_freq} from {len(text_cols)} column(s) ({len(rare_words)} rare words removed)",
    )
