"""handle_word_frequency handler."""
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


def handle_word_frequency(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Analyze word frequency — returns top-N words + Plotly bar chart."""
    col = params.get("column")
    n = int(params.get("n", 20))
    remove_stops = params.get("remove_stopwords", True)
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    corpus = df[target].fillna("").astype(str).str.lower()
    all_words: list[str] = []
    for text in corpus:
        words = re.findall(r"\b\w+\b", text)
        if remove_stops:
            words = [w for w in words if w not in ENGLISH_STOPWORDS]
        all_words.extend(words)

    freq = Counter(all_words).most_common(n)
    freq_df = pd.DataFrame(freq, columns=["word", "count"])
    freq_df["percentage"] = (freq_df["count"] / max(len(all_words), 1) * 100).round(2)

    fig = px.bar(
        freq_df, x="count", y="word", orientation="h",
        text="count",
    )
    fig.update_traces(marker_color="#FB8C3C", textposition="outside")
    _style(fig, title=f"Top {n} Words — {target} ({len(all_words):,} total words)")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Frequency", yaxis_title="Word")

    return HandlerResult(
        success=True, result_df=freq_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Top {n} words from '{target}' ({len(all_words):,} total, {len(set(all_words)):,} unique)",
    )
