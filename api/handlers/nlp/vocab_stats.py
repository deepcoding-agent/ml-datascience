"""handle_vocab_stats handler."""
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


def handle_vocab_stats(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Vocabulary statistics: unique tokens, type-token ratio, avg word length,
    hapax legomena (words appearing only once), top words."""
    col = params.get("column")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    rows: list[dict] = []
    for c in text_cols:
        corpus = df[c].fillna("").astype(str).str.lower()
        all_words: list[str] = []
        for text in corpus:
            all_words.extend(re.findall(r"\b\w+\b", text))

        total = len(all_words)
        unique = len(set(all_words))
        freq = Counter(all_words)
        hapax = sum(1 for w, cnt in freq.items() if cnt == 1)
        avg_len = np.mean([len(w) for w in all_words]) if all_words else 0

        rows.append({
            "column": c,
            "total_tokens": total,
            "unique_tokens": unique,
            "type_token_ratio": round(unique / max(total, 1), 4),
            "avg_word_length": round(avg_len, 2),
            "hapax_legomena": hapax,
            "hapax_ratio": round(hapax / max(unique, 1), 4),
            "top_5_words": ", ".join(w for w, _ in freq.most_common(5)),
            "docs_with_text": int((corpus.str.len() > 0).sum()),
            "empty_docs": int((corpus.str.len() == 0).sum()),
        })

    result_df = pd.DataFrame(rows)
    lines = []
    for r in rows:
        lines.append(
            f"**{r['column']}**: {r['total_tokens']:,} tokens, "
            f"{r['unique_tokens']:,} unique (TTR={r['type_token_ratio']}), "
            f"avg length={r['avg_word_length']}"
        )

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        summary="\n".join(lines),
    )
