"""handle_sentence_features handler."""
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


def handle_sentence_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract sentence-level features: sentence count, avg/min/max sentence length,
    question/exclamation counts."""
    col = params.get("column")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    created: list[str] = []
    for c in text_cols:
        s = result[c].fillna("").astype(str)

        def sent_stats(text: str) -> dict:
            sents = [x.strip() for x in re.split(r"[.!?]+", text) if x.strip()]
            n = len(sents)
            lens = [len(x.split()) for x in sents] if sents else [0]
            return {
                "count": n,
                "avg_len": round(np.mean(lens), 2),
                "max_len": max(lens),
                "min_len": min(lens),
            }

        stats = s.apply(sent_stats).apply(pd.Series)
        result[f"{c}_sent_count"] = stats["count"].astype(int)
        result[f"{c}_sent_avg_len"] = stats["avg_len"]
        result[f"{c}_sent_max_len"] = stats["max_len"].astype(int)
        result[f"{c}_sent_min_len"] = stats["min_len"].astype(int)
        result[f"{c}_question_count"] = s.str.count(r"\?")
        result[f"{c}_exclamation_count"] = s.str.count(r"!")

        created.extend([
            f"{c}_sent_count", f"{c}_sent_avg_len", f"{c}_sent_max_len",
            f"{c}_sent_min_len", f"{c}_question_count", f"{c}_exclamation_count",
        ])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Extracted {len(created)} sentence features from {len(text_cols)} column(s)",
    )
