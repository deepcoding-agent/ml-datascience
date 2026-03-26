"""handle_regex_extract handler."""
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


def handle_regex_extract(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract patterns from text: email, url, hashtag, mention, phone, number, or custom regex."""
    col = params.get("column")
    pattern_name = params.get("pattern", "all")  # email|url|hashtag|mention|phone|number|all|custom
    custom_regex = params.get("regex")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    patterns: dict[str, str] = {}
    if custom_regex:
        patterns["custom"] = custom_regex
    elif pattern_name == "all":
        patterns = dict(_PATTERNS)
    elif pattern_name in _PATTERNS:
        patterns[pattern_name] = _PATTERNS[pattern_name]
    else:
        return HandlerResult(
            success=False,
            error=f"Unknown pattern '{pattern_name}'. Available: {list(_PATTERNS.keys())} or 'all'",
        )

    created: list[str] = []
    for c in text_cols:
        s = result[c].fillna("").astype(str)
        for pname, regex in patterns.items():
            count_col = f"{c}_{pname}_count"
            result[count_col] = s.str.count(regex)
            match_col = f"{c}_{pname}_found"
            result[match_col] = s.str.findall(regex).apply(
                lambda x: ",".join(x[:5]) if x else ""
            )
            created.extend([count_col, match_col])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Regex extract: {len(created)} columns from {len(text_cols)} text column(s)",
    )
