"""handle_text_summary_report handler."""
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


def handle_text_summary_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Comprehensive text dataset report: language mix, length stats,
    vocabulary, quality issues, and recommendations."""
    col = params.get("column")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    target = text_cols[0]
    s = df[target].fillna("").astype(str)
    findings: list[dict] = []

    # Basic stats
    empty = int((s.str.strip() == "").sum())
    word_counts = s.str.split().str.len().fillna(0)
    char_counts = s.str.len()
    findings.append({"category": "Size", "detail": f"{len(s)} documents, {int(word_counts.sum()):,} total words"})
    findings.append({"category": "Length", "detail": f"avg {word_counts.mean():.0f} words/doc, range [{int(word_counts.min())}-{int(word_counts.max())}]"})

    if empty > 0:
        findings.append({"category": "Quality", "detail": f"{empty} empty rows ({empty/len(s)*100:.1f}%)"})

    # Vocabulary
    all_words: list[str] = []
    for text in s.str.lower():
        all_words.extend(re.findall(r"\b\w+\b", text))
    vocab_size = len(set(all_words))
    findings.append({"category": "Vocabulary", "detail": f"{vocab_size:,} unique words, TTR={vocab_size/max(len(all_words),1):.4f}"})

    # Duplicates
    dup_count = s.str.lower().str.strip().duplicated().sum()
    if dup_count > 0:
        findings.append({"category": "Quality", "detail": f"{int(dup_count)} exact duplicates ({dup_count/len(s)*100:.1f}%)"})

    # Language detection sample
    def detect_script(text: str) -> str:
        for ch in text[:100]:
            cp = ord(ch)
            if 0x0E00 <= cp <= 0x0E7F: return "Thai"
            if 0x4E00 <= cp <= 0x9FFF: return "Chinese"
            if 0x3040 <= cp <= 0x30FF: return "Japanese"
        return "Latin"

    scripts = s.head(100).apply(detect_script).value_counts()
    findings.append({"category": "Language", "detail": ", ".join(f"{k}: {v}" for k, v in scripts.items())})

    # Short texts
    short = int((word_counts < 5).sum())
    if short > 0:
        findings.append({"category": "Quality", "detail": f"{short} very short texts (<5 words)"})

    # Recommendations
    recs: list[str] = []
    if empty > 0:
        recs.append(f"Remove {empty} empty rows with nlp.text_filter")
    if dup_count > 0:
        recs.append(f"Remove {int(dup_count)} duplicates with nlp.text_dedup_exact")
    if word_counts.max() > 500:
        recs.append(f"Consider nlp.text_chunk for long documents (max={int(word_counts.max())} words)")
    if vocab_size > 10000:
        recs.append("Large vocabulary — consider nlp.text_normalize or nlp.remove_stopwords")

    for rec in recs:
        findings.append({"category": "Recommendation", "detail": rec})

    result_df = pd.DataFrame(findings)

    # Chart
    fig = px.histogram(x=word_counts, nbins=30)
    fig.update_traces(marker_color="#FB8C3C")
    _style(fig, title=f"Text Report — {target}: {len(s)} docs, {vocab_size:,} vocab")
    fig.update_layout(xaxis_title="Words per document", yaxis_title="Count")

    summary = "\n".join(f"• [{r['category']}] {r['detail']}" for r in findings)
    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=summary,
    )
