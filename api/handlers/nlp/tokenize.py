"""handle_tokenize handler."""
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


def handle_tokenize(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Tokenize text into words using regex word boundaries.
    Creates {col}_tokens (list) and {col}_token_count columns."""
    col = params.get("column")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    created: list[str] = []
    for c in text_cols:
        s = result[c].fillna("").astype(str)
        tokens = s.str.findall(r"\b\w+\b")
        result[f"{c}_tokens"] = tokens.apply(lambda x: ",".join(x) if x else "")
        result[f"{c}_token_count"] = tokens.str.len().fillna(0).astype(int)
        created.extend([f"{c}_tokens", f"{c}_token_count"])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Tokenized {len(text_cols)} column(s) → {len(created)} new columns",
    )
