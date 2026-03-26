"""handle_tfidf handler."""
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


def handle_tfidf(df: pd.DataFrame, params: dict) -> HandlerResult:
    """TF-IDF vectorization — creates top-N feature columns from text."""
    col = params.get("column")
    max_features = int(params.get("n", 50))
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        total_features = 0
        for c in text_cols:
            corpus = result[c].fillna("").astype(str)
            vec = TfidfVectorizer(max_features=max_features, stop_words="english")
            matrix = vec.fit_transform(corpus)
            feature_names = [f"{c}_tfidf_{w}" for w in vec.get_feature_names_out()]
            tfidf_df = pd.DataFrame(
                matrix.toarray(), columns=feature_names, index=result.index,
            )
            result = pd.concat([result, tfidf_df], axis=1)
            total_features += len(feature_names)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"TF-IDF: created {total_features} features from {len(text_cols)} column(s)",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"TF-IDF error: {e}")
