"""handle_text_encode handler."""
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


def handle_text_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Encode text as integer sequences (word → ID mapping).
    Creates {col}_encoded column with comma-separated word IDs
    and {col}_vocab_size with vocabulary size. Useful for deep learning input."""
    col = params.get("column")
    max_vocab = int(params.get("max_vocab", 5000))
    max_len = int(params.get("max_len", 100))
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    created: list[str] = []
    for c in text_cols:
        corpus = result[c].fillna("").astype(str).str.lower()
        # Build vocabulary from all text
        all_words: list[str] = []
        for text in corpus:
            all_words.extend(re.findall(r"\b\w+\b", text))

        freq = Counter(all_words)
        # Reserve 0=pad, 1=unknown
        vocab = {word: idx + 2 for idx, (word, _) in enumerate(freq.most_common(max_vocab))}

        def encode_text(text: str) -> str:
            words = re.findall(r"\b\w+\b", text.lower())[:max_len]
            ids = [str(vocab.get(w, 1)) for w in words]
            # Pad to max_len
            ids.extend(["0"] * (max_len - len(ids)))
            return ",".join(ids)

        result[f"{c}_encoded"] = corpus.apply(encode_text)
        result[f"{c}_vocab_size"] = len(vocab) + 2  # +2 for pad and unknown
        created.extend([f"{c}_encoded", f"{c}_vocab_size"])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=(
            f"Encoded {len(text_cols)} column(s) as integer sequences "
            f"(vocab={min(len(freq), max_vocab)+2}, max_len={max_len})"
        ),
        metadata={"vocab_size": min(len(freq), max_vocab) + 2, "max_len": max_len},
    )
