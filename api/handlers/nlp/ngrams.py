"""handle_ngrams handler."""
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


def handle_ngrams(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract word n-gram features using TF-IDF with n-gram range."""
    col = params.get("column")
    n = int(params.get("n", 2))  # bigrams by default
    max_features = int(params.get("max_features", 30))
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        total_features = 0
        for c in text_cols:
            corpus = result[c].fillna("").astype(str)
            vec = TfidfVectorizer(
                ngram_range=(n, n), max_features=max_features, stop_words="english",
            )
            matrix = vec.fit_transform(corpus)
            names = [f"{c}_{n}gram_{w.replace(' ', '_')}" for w in vec.get_feature_names_out()]
            ngram_df = pd.DataFrame(
                matrix.toarray(), columns=names, index=result.index,
            )
            result = pd.concat([result, ngram_df], axis=1)
            total_features += len(names)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"{n}-gram features: created {total_features} from {len(text_cols)} column(s)",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"N-gram error: {e}")
