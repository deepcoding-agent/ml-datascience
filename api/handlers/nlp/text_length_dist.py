"""handle_text_length_dist handler."""
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


def handle_text_length_dist(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Analyze text length distribution (char & word count) with charts.
    Useful for choosing chunk size, max_len, or detecting anomalies."""
    col = params.get("column")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    s = df[target].fillna("").astype(str)
    char_lens = s.str.len()
    word_lens = s.str.split().str.len().fillna(0).astype(int)

    stats = pd.DataFrame({
        "metric": ["count", "mean_chars", "median_chars", "std_chars", "min_chars", "max_chars",
                    "mean_words", "median_words", "min_words", "max_words",
                    "empty_rows", "single_word_rows"],
        "value": [len(s), round(float(char_lens.mean()), 1), int(char_lens.median()),
                  round(float(char_lens.std()), 1), int(char_lens.min()), int(char_lens.max()),
                  round(float(word_lens.mean()), 1), int(word_lens.median()),
                  int(word_lens.min()), int(word_lens.max()),
                  int((char_lens == 0).sum()), int((word_lens <= 1).sum())],
    })

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go_fig
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Character Length", "Word Count"])
    fig.add_trace(go_fig.Histogram(x=char_lens, nbinsx=30, marker_color="#FB8C3C", name="Chars"), row=1, col=1)
    fig.add_trace(go_fig.Histogram(x=word_lens, nbinsx=30, marker_color="#2EC4B6", name="Words"), row=1, col=2)
    _style(fig, title=f"Text Length Distribution — {target} (n={len(s)})")
    fig.update_layout(showlegend=False, height=350)

    return HandlerResult(
        success=True, result_df=stats, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"'{target}': avg {word_lens.mean():.0f} words/doc, range [{word_lens.min()}-{word_lens.max()}], {(char_lens==0).sum()} empty",
    )
