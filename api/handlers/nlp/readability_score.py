"""handle_readability_score handler."""
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


def handle_readability_score(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compute readability metrics: Flesch Reading Ease (approx),
    Coleman-Liau Index, Automated Readability Index (ARI)."""
    col = params.get("column")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    def _count_syllables(word: str) -> int:
        """Approximate syllable count using vowel groups."""
        word = word.lower().rstrip("e")
        vowels = re.findall(r"[aeiouy]+", word)
        return max(1, len(vowels))

    created: list[str] = []
    for c in text_cols:
        s = result[c].fillna("").astype(str)

        def readability(text: str) -> dict:
            words_list = re.findall(r"\b\w+\b", text)
            sents = [x for x in re.split(r"[.!?]+", text) if x.strip()]
            n_words = max(len(words_list), 1)
            n_sents = max(len(sents), 1)
            n_chars = sum(len(w) for w in words_list)
            n_syllables = sum(_count_syllables(w) for w in words_list)
            # Flesch Reading Ease
            fre = 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (n_syllables / n_words)
            # Coleman-Liau Index
            L = (n_chars / n_words) * 100
            S = (n_sents / n_words) * 100
            cli = 0.0588 * L - 0.296 * S - 15.8
            # Automated Readability Index
            ari = 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sents) - 21.43
            return {
                "flesch": round(max(0, min(fre, 120)), 1),
                "coleman_liau": round(cli, 1),
                "ari": round(ari, 1),
            }

        scores = s.apply(readability).apply(pd.Series)
        result[f"{c}_flesch_score"] = scores["flesch"]
        result[f"{c}_coleman_liau"] = scores["coleman_liau"]
        result[f"{c}_ari_score"] = scores["ari"]
        created.extend([f"{c}_flesch_score", f"{c}_coleman_liau", f"{c}_ari_score"])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Computed readability scores for {len(text_cols)} column(s): Flesch, Coleman-Liau, ARI",
    )
