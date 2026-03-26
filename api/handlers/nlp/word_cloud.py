"""handle_word_cloud handler."""
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


def handle_word_cloud(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Generate word cloud data (word + weight) with Plotly treemap visualization."""
    col = params.get("column")
    n = int(params.get("n", 40))
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    corpus = df[target].fillna("").astype(str).str.lower()
    all_words: list[str] = []
    for text in corpus:
        words = [w for w in re.findall(r"\b\w+\b", text) if w not in ENGLISH_STOPWORDS and len(w) > 2]
        all_words.extend(words)

    freq = Counter(all_words).most_common(n)
    if not freq:
        return HandlerResult(success=False, error="No words found after filtering")

    cloud_df = pd.DataFrame(freq, columns=["word", "weight"])
    cloud_df["percentage"] = (cloud_df["weight"] / max(sum(c for _, c in freq), 1) * 100).round(2)

    fig = px.treemap(
        cloud_df, path=["word"], values="weight",
        color="weight", color_continuous_scale="YlOrRd",
    )
    _style(fig, title=f"Word Cloud — {target} (top {n} words)")
    fig.update_traces(textinfo="label+value")

    return HandlerResult(
        success=True, result_df=cloud_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"Word cloud for '{target}': top {n} words from {len(all_words):,} total",
    )
