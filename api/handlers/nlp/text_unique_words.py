"""handle_text_unique_words handler."""
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


def handle_text_unique_words(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract words that appear only in one document (unique to that doc).
    Creates a column with comma-separated rare/unique words per row."""
    col = params.get("column")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    corpus = result[target].fillna("").astype(str).str.lower()

    # Build document frequency
    doc_freq: Counter = Counter()
    doc_words: list[set] = []
    for text in corpus:
        words = set(re.findall(r"\b\w+\b", text)) - ENGLISH_STOPWORDS
        doc_words.append(words)
        doc_freq.update(words)

    # Words appearing in only 1 document
    unique_to_doc: list[str] = []
    for words in doc_words:
        uniques = sorted(w for w in words if doc_freq[w] == 1)
        unique_to_doc.append(", ".join(uniques[:10]))

    result[f"{target}_unique_words"] = unique_to_doc
    result[f"{target}_unique_count"] = [len(w.split(", ")) if w else 0 for w in unique_to_doc]

    total_unique = sum(1 for _, cnt in doc_freq.items() if cnt == 1)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Found {total_unique} corpus-unique words across {len(corpus)} documents",
    )
