"""handle_word_overlap handler."""
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


def handle_word_overlap(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compute word overlap (Jaccard similarity) between two text columns
    or between consecutive rows of the same column."""
    columns = params.get("columns", [])
    col = params.get("column")
    result = df.copy()

    if columns and len(columns) >= 2 and all(c in df.columns for c in columns[:2]):
        c1, c2 = columns[0], columns[1]
        s1 = result[c1].fillna("").astype(str).str.lower()
        s2 = result[c2].fillna("").astype(str).str.lower()

        def jaccard(a: str, b: str) -> float:
            wa = set(re.findall(r"\b\w+\b", a))
            wb = set(re.findall(r"\b\w+\b", b))
            if not wa and not wb:
                return 0.0
            return round(len(wa & wb) / max(len(wa | wb), 1), 4)

        result[f"{c1}_{c2}_jaccard"] = [jaccard(a, b) for a, b in zip(s1, s2)]
        avg = result[f"{c1}_{c2}_jaccard"].mean()
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Jaccard similarity between '{c1}' and '{c2}': avg={avg:.4f}",
        )
    else:
        # Consecutive row overlap within single column
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found. Provide columns=[col1, col2] or column=col")
        target = text_cols[0]
        s = result[target].fillna("").astype(str).str.lower()
        overlaps: list[float] = [0.0]
        for i in range(1, len(s)):
            wa = set(re.findall(r"\b\w+\b", s.iloc[i - 1]))
            wb = set(re.findall(r"\b\w+\b", s.iloc[i]))
            j = len(wa & wb) / max(len(wa | wb), 1) if (wa or wb) else 0.0
            overlaps.append(round(j, 4))
        result[f"{target}_row_overlap"] = overlaps
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Row-to-row Jaccard overlap for '{target}': avg={np.mean(overlaps):.4f}",
        )
