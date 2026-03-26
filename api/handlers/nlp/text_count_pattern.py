"""handle_text_count_pattern handler."""
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


def handle_text_count_pattern(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Count occurrences of a specific pattern (word/phrase/regex) per row.
    Creates a count column and optionally filters rows with matches."""
    col = params.get("column")
    pattern = params.get("pattern", "")
    filter_matches = params.get("filter", False)
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")
    if not pattern:
        return HandlerResult(success=False, error="Specify pattern= parameter")

    target = text_cols[0]
    result = df.copy()
    s = result[target].fillna("").astype(str)
    try:
        result[f"count_{pattern[:20]}"] = s.str.count(pattern)
    except re.error:
        result[f"count_{pattern[:20]}"] = s.str.count(re.escape(pattern))

    total = int(result[f"count_{pattern[:20]}"].sum())
    rows_with = int((result[f"count_{pattern[:20]}"] > 0).sum())

    if filter_matches:
        result = result[result[f"count_{pattern[:20]}"] > 0].reset_index(drop=True)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Pattern '{pattern}': {total} occurrences in {rows_with}/{len(df)} rows",
    )
