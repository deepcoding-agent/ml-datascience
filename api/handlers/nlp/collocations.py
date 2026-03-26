"""handle_collocations handler."""
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


def handle_collocations(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Find significant word collocations (bigrams) using frequency and PMI.
    Returns top-N collocations with scores + bar chart."""
    col = params.get("column")
    n = int(params.get("n", 20))
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    corpus = df[target].fillna("").astype(str).str.lower()

    # Collect bigrams and unigram counts
    bigram_counts: Counter = Counter()
    unigram_counts: Counter = Counter()
    total_bigrams = 0

    for text in corpus:
        words = [w for w in re.findall(r"\b\w+\b", text) if w not in ENGLISH_STOPWORDS and len(w) > 1]
        unigram_counts.update(words)
        for i in range(len(words) - 1):
            bigram_counts[(words[i], words[i + 1])] += 1
            total_bigrams += 1

    if total_bigrams == 0:
        return HandlerResult(success=False, error="No collocations found — text may be too short")

    total_unigrams = sum(unigram_counts.values())

    # Compute PMI for top bigrams
    rows: list[dict] = []
    for (w1, w2), count in bigram_counts.most_common(n * 3):
        if count < 2:
            continue
        p_bi = count / total_bigrams
        p_w1 = unigram_counts[w1] / total_unigrams
        p_w2 = unigram_counts[w2] / total_unigrams
        pmi = np.log2(p_bi / max(p_w1 * p_w2, 1e-10))
        rows.append({
            "collocation": f"{w1} {w2}",
            "frequency": count,
            "pmi": round(pmi, 3),
        })

    rows.sort(key=lambda r: r["pmi"], reverse=True)
    result_df = pd.DataFrame(rows[:n])

    if not result_df.empty:
        fig = px.bar(
            result_df, x="pmi", y="collocation", orientation="h",
            text="frequency",
        )
        fig.update_traces(marker_color="#2EC4B6", textposition="outside")
        _style(fig, title=f"Top {n} Collocations — {target} (PMI-ranked)")
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="PMI Score", yaxis_title="Word Pair",
        )
        charts = [fig.to_json()]
    else:
        charts = []

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=charts,
        summary=f"Found {len(result_df)} significant collocations in '{target}'",
    )
