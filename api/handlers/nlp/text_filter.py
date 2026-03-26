"""handle_text_filter handler."""
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


def handle_text_filter(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Filter rows by text criteria: min/max length, contains keyword,
    min word count. Useful for cleaning out empty or too-short texts."""
    col = params.get("column")
    min_len = params.get("min_len")          # minimum character length
    max_len = params.get("max_len")          # maximum character length
    min_words = params.get("min_words")      # minimum word count
    contains = params.get("contains")        # must contain keyword
    not_contains = params.get("not_contains") # must not contain keyword
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    result = df.copy()
    s = result[target].fillna("").astype(str)
    original = len(result)
    mask = pd.Series(True, index=result.index)

    if min_len is not None:
        mask &= s.str.len() >= int(min_len)
    if max_len is not None:
        mask &= s.str.len() <= int(max_len)
    if min_words is not None:
        mask &= s.str.split().str.len().fillna(0) >= int(min_words)
    if contains is not None:
        mask &= s.str.contains(str(contains), case=False, na=False)
    if not_contains is not None:
        mask &= ~s.str.contains(str(not_contains), case=False, na=False)

    result = result[mask].reset_index(drop=True)
    removed = original - len(result)

    filters = []
    if min_len is not None: filters.append(f"min_len={min_len}")
    if max_len is not None: filters.append(f"max_len={max_len}")
    if min_words is not None: filters.append(f"min_words={min_words}")
    if contains is not None: filters.append(f"contains='{contains}'")
    if not_contains is not None: filters.append(f"not_contains='{not_contains}'")

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Filtered '{target}': {original} → {len(result)} rows (removed {removed}), filters: {', '.join(filters)}",
    )
