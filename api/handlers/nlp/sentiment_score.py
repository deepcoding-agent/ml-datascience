"""handle_sentiment_score handler."""
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


def handle_sentiment_score(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Basic lexicon-based sentiment scoring (positive/negative/compound).
    Uses built-in word lists — no external NLP library needed."""
    col = params.get("column")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    created: list[str] = []
    for c in text_cols:
        s = result[c].fillna("").astype(str).str.lower()
        words_series = s.str.findall(r"\b\w+\b")

        pos_counts = words_series.apply(lambda ws: sum(1 for w in ws if w in _POSITIVE_WORDS))
        neg_counts = words_series.apply(lambda ws: sum(1 for w in ws if w in _NEGATIVE_WORDS))
        total = words_series.str.len().replace(0, 1)

        result[f"{c}_sentiment_pos"] = (pos_counts / total).round(4)
        result[f"{c}_sentiment_neg"] = (neg_counts / total).round(4)
        result[f"{c}_sentiment_compound"] = ((pos_counts - neg_counts) / total).round(4)
        result[f"{c}_sentiment_label"] = np.where(
            result[f"{c}_sentiment_compound"] > 0.05, "positive",
            np.where(result[f"{c}_sentiment_compound"] < -0.05, "negative", "neutral"),
        )
        created.extend([
            f"{c}_sentiment_pos", f"{c}_sentiment_neg",
            f"{c}_sentiment_compound", f"{c}_sentiment_label",
        ])

    # Summary chart for first column
    charts: list[str] = []
    if text_cols:
        label_col = f"{text_cols[0]}_sentiment_label"
        if label_col in result.columns:
            counts = result[label_col].value_counts()
            fig = px.bar(
                x=counts.index, y=counts.values,
                color=counts.index,
                color_discrete_map={"positive": "#2EC4B6", "neutral": "#86868B", "negative": "#E71D36"},
            )
            _style(fig, title=f"Sentiment Distribution — {text_cols[0]} (n={len(result)})")
            fig.update_layout(xaxis_title="Sentiment", yaxis_title="Count", showlegend=False)
            charts.append(fig.to_json())

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        charts_plotly=charts,
        summary=f"Sentiment scored {len(text_cols)} column(s): {len(created)} features created",
    )
