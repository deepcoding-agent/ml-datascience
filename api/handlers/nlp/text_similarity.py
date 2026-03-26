"""handle_text_similarity handler."""
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


def handle_text_similarity(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compute pairwise text similarity using TF-IDF + cosine similarity.
    Returns similarity matrix (first 50 rows for performance)."""
    col = params.get("column")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        target = text_cols[0]
        sample = df.head(50)
        corpus = sample[target].fillna("").astype(str)
        vec = TfidfVectorizer(max_features=100, stop_words="english")
        matrix = vec.fit_transform(corpus)
        sim = cosine_similarity(matrix)
        sim_df = pd.DataFrame(
            sim.round(3),
            index=[f"doc_{i}" for i in range(len(sim))],
            columns=[f"doc_{i}" for i in range(len(sim))],
        )

        import plotly.graph_objects as go
        fig = go.Figure(data=go.Heatmap(
            z=sim, colorscale="YlOrRd",
            x=sim_df.columns.tolist(), y=sim_df.index.tolist(),
        ))
        _style(fig, title=f"Text Similarity (cosine) — {target} (n={len(sample)})")

        return HandlerResult(
            success=True, result_df=sim_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Cosine similarity matrix for '{target}' ({len(sample)} documents)",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Text similarity error: {e}")
