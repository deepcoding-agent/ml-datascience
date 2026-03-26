"""handle_text_chunk handler."""
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


def handle_text_chunk(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Split long texts into fixed-size chunks (by word count).
    Creates new rows for each chunk — useful for processing long documents."""
    col = params.get("column")
    chunk_size = int(params.get("chunk_size", 200))  # words per chunk
    overlap = int(params.get("overlap", 20))
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    chunks: list[dict] = []

    for idx, row in df.iterrows():
        text = str(row.get(target, ""))
        words = text.split()

        if len(words) <= chunk_size:
            chunk_row = row.to_dict()
            chunk_row["_chunk_id"] = 0
            chunk_row["_chunk_total"] = 1
            chunks.append(chunk_row)
        else:
            step = max(chunk_size - overlap, 1)
            total_chunks = max(1, (len(words) - overlap) // step + (1 if (len(words) - overlap) % step else 0))
            for i, start in enumerate(range(0, len(words), step)):
                chunk_words = words[start : start + chunk_size]
                if not chunk_words:
                    break
                chunk_row = row.to_dict()
                chunk_row[target] = " ".join(chunk_words)
                chunk_row["_chunk_id"] = i
                chunk_row["_chunk_total"] = total_chunks
                chunks.append(chunk_row)

    result = pd.DataFrame(chunks).reset_index(drop=True)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Chunked '{target}': {len(df)} docs → {len(result)} chunks (size={chunk_size}, overlap={overlap})",
    )
