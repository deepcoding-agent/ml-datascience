"""handle_text_to_paragraphs handler."""
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


def handle_text_to_paragraphs(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Split text into paragraphs (by blank lines). Each paragraph becomes a new row."""
    col = params.get("column")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    rows: list[dict] = []

    for _, row in df.iterrows():
        text = str(row.get(target, ""))
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        for i, para in enumerate(paragraphs):
            new_row = row.to_dict()
            new_row[target] = para
            new_row["_para_id"] = i
            new_row["_para_total"] = len(paragraphs)
            rows.append(new_row)

    result = pd.DataFrame(rows).reset_index(drop=True)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Split '{target}': {len(df)} docs → {len(result)} paragraphs",
    )
