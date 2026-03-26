"""handle_text_split_sentences handler."""
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


def handle_text_split_sentences(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Split text into individual sentences. Each sentence becomes a new row.
    Useful for sentence-level classification or analysis."""
    col = params.get("column")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    rows: list[dict] = []

    for _, row in df.iterrows():
        text = str(row.get(target, ""))
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sents:
            sents = [text]
        for i, sent in enumerate(sents):
            new_row = row.to_dict()
            new_row[target] = sent
            new_row["_sentence_id"] = i
            new_row["_sentence_total"] = len(sents)
            rows.append(new_row)

    result = pd.DataFrame(rows).reset_index(drop=True)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Split '{target}': {len(df)} docs → {len(result)} sentences",
    )
