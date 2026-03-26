"""handle_spelling_features handler."""
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


def handle_spelling_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Estimate out-of-vocabulary (OOV) word ratio as a proxy for spelling quality.
    Builds a vocabulary from the corpus; words appearing only once are likely misspelled.
    Creates oov_count and oov_ratio columns."""
    col = params.get("column")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    created: list[str] = []
    for c in text_cols:
        corpus = result[c].fillna("").astype(str).str.lower()
        # Build corpus vocabulary
        all_words: list[str] = []
        for text in corpus:
            all_words.extend(re.findall(r"\b[a-zA-Z]+\b", text))

        freq = Counter(all_words)
        # Words appearing only once with length > 2 are likely misspelled
        rare = {w for w, cnt in freq.items() if cnt == 1 and len(w) > 2}

        def oov_stats(text: str) -> tuple[int, float]:
            words = [w for w in re.findall(r"\b[a-zA-Z]+\b", text.lower()) if len(w) > 2]
            if not words:
                return 0, 0.0
            oov = sum(1 for w in words if w in rare)
            return oov, round(oov / len(words), 4)

        stats = corpus.apply(oov_stats)
        result[f"{c}_oov_count"] = stats.apply(lambda x: x[0])
        result[f"{c}_oov_ratio"] = stats.apply(lambda x: x[1])
        created.extend([f"{c}_oov_count", f"{c}_oov_ratio"])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Computed OOV/spelling features for {len(text_cols)} column(s): {len(rare)} rare words detected",
    )
