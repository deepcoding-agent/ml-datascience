"""handle_keyword_extract handler."""
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


def handle_keyword_extract(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract top keywords per document using TF-IDF scores.
    Creates {col}_keywords column with comma-separated top words."""
    col = params.get("column")
    n_keywords = int(params.get("n", 5))
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        created: list[str] = []
        for c in text_cols:
            corpus = result[c].fillna("").astype(str)
            vec = TfidfVectorizer(max_features=500, stop_words="english")
            matrix = vec.fit_transform(corpus)
            feature_names = vec.get_feature_names_out()

            keywords: list[str] = []
            for i in range(matrix.shape[0]):
                row = matrix[i].toarray().flatten()
                top_idx = row.argsort()[-n_keywords:][::-1]
                top_words = [feature_names[j] for j in top_idx if row[j] > 0]
                keywords.append(", ".join(top_words))

            result[f"{c}_keywords"] = keywords
            created.append(f"{c}_keywords")

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Extracted top-{n_keywords} keywords from {len(text_cols)} column(s)",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Keyword extraction error: {e}")
