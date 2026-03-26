"""handle_text_dedup handler."""
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


def handle_text_dedup(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Find and remove near-duplicate texts using TF-IDF cosine similarity.
    Rows with similarity > threshold are flagged or removed."""
    col = params.get("column")
    threshold = float(params.get("threshold", 0.9))
    action = params.get("action", "flag")  # flag | remove
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        target = text_cols[0]
        result = df.copy()
        corpus = result[target].fillna("").astype(str)
        vec = TfidfVectorizer(max_features=200, stop_words="english")
        matrix = vec.fit_transform(corpus)
        sim = cosine_similarity(matrix)

        # Find duplicates (keep first occurrence)
        is_dup = np.zeros(len(result), dtype=bool)
        dup_of: list[int] = [-1] * len(result)
        for i in range(len(sim)):
            if is_dup[i]:
                continue
            for j in range(i + 1, len(sim)):
                if not is_dup[j] and sim[i][j] >= threshold:
                    is_dup[j] = True
                    dup_of[j] = i

        n_dups = int(is_dup.sum())
        result["_text_is_duplicate"] = is_dup
        result["_text_duplicate_of"] = dup_of

        if action == "remove" and n_dups > 0:
            result = result[~result["_text_is_duplicate"]].drop(
                columns=["_text_is_duplicate", "_text_duplicate_of"],
            ).reset_index(drop=True)
            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Removed {n_dups} near-duplicate texts (threshold={threshold}), {len(result)} rows remain",
            )

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Flagged {n_dups} near-duplicate texts out of {len(result)} (threshold={threshold})",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Text dedup error: {e}")
