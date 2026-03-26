"""handle_text_augment handler."""
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


def handle_text_augment(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Simple text augmentation: random word deletion, swap, and synonym-free duplication.
    Creates augmented copies appended to the dataset. Useful for balancing NLP training sets."""
    col = params.get("column")
    n_aug = int(params.get("n", 1))  # number of augmented copies per row
    strategy = params.get("strategy", "mixed")  # delete | swap | mixed
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    rng = np.random.RandomState(42)
    target = text_cols[0]

    def augment_one(text: str, method: str) -> str:
        words = text.split()
        if len(words) < 3:
            return text
        if method == "delete":
            idx = rng.randint(0, len(words))
            return " ".join(w for i, w in enumerate(words) if i != idx)
        elif method == "swap":
            i = rng.randint(0, max(len(words) - 1, 1))
            j = min(i + 1, len(words) - 1)
            words[i], words[j] = words[j], words[i]
            return " ".join(words)
        else:  # mixed
            method = rng.choice(["delete", "swap"])
            return augment_one(text, method)

    result = df.copy()
    aug_rows: list[pd.DataFrame] = []
    for _ in range(n_aug):
        aug = result.copy()
        aug[target] = aug[target].fillna("").astype(str).apply(
            lambda t: augment_one(t, strategy)
        )
        aug_rows.append(aug)

    result = pd.concat([result] + aug_rows, ignore_index=True)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Augmented '{target}': {len(df)} → {len(result)} rows ({n_aug}x augmentation, strategy={strategy})",
    )
