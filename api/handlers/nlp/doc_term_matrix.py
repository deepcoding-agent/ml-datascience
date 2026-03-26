"""handle_doc_term_matrix handler."""
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


def handle_doc_term_matrix(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Build a full document-term frequency matrix.
    Returns a sparse-style DataFrame with word counts per document."""
    col = params.get("column")
    max_features = int(params.get("n", 100))
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        target = text_cols[0]
        corpus = df[target].fillna("").astype(str)
        vec = CountVectorizer(max_features=max_features, stop_words="english")
        matrix = vec.fit_transform(corpus)
        dtm = pd.DataFrame(
            matrix.toarray(),
            columns=vec.get_feature_names_out(),
            index=df.index,
        )
        return HandlerResult(
            success=True, result_df=dtm, output_type="generate",
            summary=f"Document-term matrix: {dtm.shape[0]} docs × {dtm.shape[1]} terms",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Doc-term matrix error: {e}")
