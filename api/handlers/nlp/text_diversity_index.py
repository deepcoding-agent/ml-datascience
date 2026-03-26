"""handle_text_diversity_index handler."""
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


def handle_text_diversity_index(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compute Simpson diversity index of words per document.
    Higher values indicate more diverse vocabulary within a text."""
    col = params.get("column")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    result = df.copy()
    created: list[str] = []
    for c in text_cols:
        s = result[c].fillna("").astype(str).str.lower()

        def _simpson(text: str) -> float:
            words = re.findall(r"\b\w+\b", text)
            n = len(words)
            if n <= 1:
                return 0.0
            freq = Counter(words)
            denom = n * (n - 1)
            numerator = sum(cnt * (cnt - 1) for cnt in freq.values())
            return round(1.0 - (numerator / denom), 4)

        result[f"{c}_diversity"] = s.apply(_simpson)
        created.append(f"{c}_diversity")

    mean_div = result[created[0]].mean()
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Simpson diversity index for {len(text_cols)} column(s) (mean={mean_div:.4f})",
    )
