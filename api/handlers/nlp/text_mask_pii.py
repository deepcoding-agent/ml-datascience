"""handle_text_mask_pii handler."""
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


def handle_text_mask_pii(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Mask personally identifiable information: emails, phone numbers,
    credit card numbers, SSN-like patterns, IP addresses."""
    col = params.get("column")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    pii_patterns: list[tuple[str, str, str]] = [
        ("email", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
        ("phone", r"\+?\d[\d\-\s()]{7,}\d", "[PHONE]"),
        ("credit_card", r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "[CREDIT_CARD]"),
        ("ssn", r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b", "[SSN]"),
        ("ip_address", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]"),
        ("url", r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", "[URL]"),
    ]

    total_masked = 0
    for c in text_cols:
        s = result[c].fillna("").astype(str)
        for pii_name, pattern, replacement in pii_patterns:
            count_before = s.str.count(pattern).sum()
            s = s.str.replace(pattern, replacement, regex=True)
            total_masked += int(count_before)
        result[c] = s

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Masked {total_masked} PII instances across {len(text_cols)} column(s): email, phone, credit card, SSN, IP, URL",
    )
