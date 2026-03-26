"""handle_text_dedup_exact handler."""
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


def handle_text_dedup_exact(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Remove rows with exact duplicate text (case-insensitive).
    Much faster than similarity-based dedup for large datasets."""
    col = params.get("column")
    keep = params.get("keep", "first")  # first | last
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    result = df.copy()
    result["_text_lower"] = result[target].fillna("").astype(str).str.lower().str.strip()
    original = len(result)
    result = result.drop_duplicates(subset="_text_lower", keep=keep).drop(columns="_text_lower").reset_index(drop=True)
    removed = original - len(result)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Exact dedup on '{target}': {original} → {len(result)} rows (removed {removed} duplicates)",
    )
