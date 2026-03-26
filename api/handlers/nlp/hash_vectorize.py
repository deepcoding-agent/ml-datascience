"""handle_hash_vectorize handler."""
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


def handle_hash_vectorize(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Feature hashing — fast, memory-efficient text vectorization.
    Creates N hashed feature columns (no vocabulary needed)."""
    col = params.get("column")
    n_features = int(params.get("n", 32))
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    try:
        from sklearn.feature_extraction.text import HashingVectorizer

        total = 0
        for c in text_cols:
            corpus = result[c].fillna("").astype(str)
            vec = HashingVectorizer(n_features=n_features, alternate_sign=False)
            matrix = vec.fit_transform(corpus)
            names = [f"{c}_hash_{i}" for i in range(n_features)]
            hash_df = pd.DataFrame(
                matrix.toarray(), columns=names, index=result.index,
            )
            result = pd.concat([result, hash_df], axis=1)
            total += n_features

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Hash vectorization: {total} features from {len(text_cols)} column(s)",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Hash vectorize error: {e}")
