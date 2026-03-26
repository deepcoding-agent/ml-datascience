"""handle_text_replace handler."""
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


def handle_text_replace(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Find and replace text patterns (regex or literal) across text columns.
    Supports multiple replacements via a mapping dict."""
    col = params.get("column")
    pattern = params.get("pattern", "")
    replacement = params.get("replacement", "")
    mapping = params.get("mapping")  # dict of {find: replace, ...}
    use_regex = params.get("regex", True)
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    total_replacements = 0
    for c in text_cols:
        s = result[c].fillna("").astype(str)
        if mapping and isinstance(mapping, dict):
            for find, repl in mapping.items():
                count = s.str.count(find).sum()
                total_replacements += int(count)
                s = s.str.replace(find, str(repl), regex=bool(use_regex))
        elif pattern:
            count = s.str.count(pattern).sum()
            total_replacements += int(count)
            s = s.str.replace(pattern, replacement, regex=bool(use_regex))
        result[c] = s

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Replaced {total_replacements} occurrences across {len(text_cols)} column(s)",
    )
