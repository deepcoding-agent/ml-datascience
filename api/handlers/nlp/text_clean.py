"""handle_text_clean handler."""
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


def handle_text_clean(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Clean text: lowercase, remove HTML/URLs/emails/punctuation/numbers,
    normalize whitespace. Use strategy param to select specific steps."""
    col = params.get("column")
    strategy = params.get("strategy", "all")  # all|lowercase|no_punct|no_numbers|no_html|no_urls|no_emails
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    steps_applied: list[str] = []
    for c in text_cols:
        s = result[c].fillna("").astype(str)

        if strategy in ("all", "no_html"):
            s = s.str.replace(r"<[^>]+>", " ", regex=True)
            if "html" not in steps_applied:
                steps_applied.append("remove_html")

        if strategy in ("all", "no_urls"):
            s = s.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
            if "urls" not in steps_applied:
                steps_applied.append("remove_urls")

        if strategy in ("all", "no_emails"):
            s = s.str.replace(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", " ", regex=True)
            if "emails" not in steps_applied:
                steps_applied.append("remove_emails")

        if strategy in ("all", "no_numbers"):
            s = s.str.replace(r"\d+", " ", regex=True)
            if "numbers" not in steps_applied:
                steps_applied.append("remove_numbers")

        if strategy in ("all", "no_punct"):
            s = s.str.replace(r"[^\w\s]", " ", regex=True)
            if "punct" not in steps_applied:
                steps_applied.append("remove_punctuation")

        if strategy in ("all", "lowercase"):
            s = s.str.lower()
            if "lowercase" not in steps_applied:
                steps_applied.append("lowercase")

        # Always normalize whitespace
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
        result[c] = s

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Cleaned {len(text_cols)} text column(s): {', '.join(steps_applied)}",
    )
