"""handle_text_ngram_frequency handler."""
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


def handle_text_ngram_frequency(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Word n-gram frequency analysis with bar chart (output_type=query)."""
    col = params.get("column")
    n = int(params.get("n", 2))
    top_k = int(params.get("top", 20))
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    all_ngrams: Counter = Counter()
    for text in df[target].fillna("").astype(str).str.lower():
        words = re.findall(r"\b\w+\b", text)
        words = [w for w in words if w not in ENGLISH_STOPWORDS]
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            all_ngrams[gram] += 1

    top = all_ngrams.most_common(top_k)
    if not top:
        return HandlerResult(success=True, summary="No n-grams found", output_type="query")

    result_df = pd.DataFrame(top, columns=["ngram", "count"])
    fig = px.bar(result_df, x="count", y="ngram", orientation="h")
    fig.update_traces(marker_color="#FB8C3C")
    _style(fig, title=f"Top {len(top)} {n}-grams — {target}")
    fig.update_layout(yaxis=dict(autorange="reversed"))

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Top {len(top)} {n}-grams from '{target}': {top[0][0]} ({top[0][1]}x), {top[1][0] if len(top)>1 else ''}",
    )
