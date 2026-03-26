"""handle_text_extract_numbers handler."""
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


def handle_text_extract_numbers(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract all numbers from text into a new column (comma-separated)."""
    col = params.get("column")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    result = df.copy()
    created: list[str] = []
    for c in text_cols:
        s = result[c].fillna("").astype(str)
        result[f"{c}_numbers"] = s.apply(
            lambda t: ",".join(re.findall(r"-?\d+\.?\d*", t))
        )
        result[f"{c}_number_count"] = s.apply(
            lambda t: len(re.findall(r"-?\d+\.?\d*", t))
        )
        created.extend([f"{c}_numbers", f"{c}_number_count"])

    total = int(result[[c for c in created if c.endswith("_number_count")]].sum().sum())
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Extracted numbers from {len(text_cols)} column(s): {total} numbers found across all rows",
    )
