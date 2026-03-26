"""handle_text_window handler."""
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


def handle_text_window(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract sliding window contexts around a target keyword.
    Creates new rows with the keyword and its surrounding context."""
    col = params.get("column")
    keyword = params.get("keyword", "")
    window_size = int(params.get("window", 5))
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")
    if not keyword:
        return HandlerResult(success=False, error="Specify keyword= parameter")

    target = text_cols[0]
    contexts: list[dict] = []
    kw_lower = keyword.lower()

    for idx, row in df.iterrows():
        text = str(row.get(target, ""))
        words = text.split()
        for i, w in enumerate(words):
            if kw_lower in w.lower():
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                ctx = " ".join(words[start:end])
                contexts.append({
                    "source_row": int(idx),  # type: ignore[arg-type]
                    "keyword": keyword,
                    "position": i,
                    "context": ctx,
                })

    if not contexts:
        return HandlerResult(success=False, error=f"Keyword '{keyword}' not found in any text")

    result = pd.DataFrame(contexts)
    return HandlerResult(
        success=True, result_df=result, output_type="query",
        summary=f"Found {len(contexts)} occurrences of '{keyword}' with ±{window_size} word window",
    )
