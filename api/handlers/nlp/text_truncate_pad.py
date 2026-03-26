"""handle_text_truncate_pad handler."""
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


def handle_text_truncate_pad(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Truncate or pad text to a fixed word count.
    Useful for preparing uniform-length input for models."""
    col = params.get("column")
    max_words = int(params.get("max_words", 128))
    pad_token = params.get("pad_token", "")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    for c in text_cols:
        def trunc_pad(text: str) -> str:
            words = text.split()[:max_words]
            if pad_token and len(words) < max_words:
                words.extend([pad_token] * (max_words - len(words)))
            return " ".join(words)

        result[c] = result[c].fillna("").astype(str).apply(trunc_pad)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Truncated/padded {len(text_cols)} column(s) to {max_words} words",
    )
