"""handle_emoji_features handler."""
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


def handle_emoji_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract emoji and emoticon features: count, ratio, list of emojis found.
    Detects Unicode emojis and common text emoticons."""
    col = params.get("column")
    result = df.copy()
    text_cols = _get_text_cols(result, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")

    # Unicode emoji pattern (covers most common emoji ranges)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"   # symbols & pictographs
        "\U0001F680-\U0001F6FF"   # transport & map
        "\U0001F1E0-\U0001F1FF"   # flags
        "\U00002702-\U000027B0"   # dingbats
        "\U000024C2-\U0001F251"   # misc
        "\U0001F900-\U0001F9FF"   # supplemental
        "\U0001FA00-\U0001FA6F"   # chess symbols
        "\U0001FA70-\U0001FAFF"   # symbols extended
        "]+", flags=re.UNICODE,
    )
    # Common text emoticons
    emoticon_pattern = re.compile(
        r"(?:[:;=][-']?[)(DPp/\\|Oo3><])|(?:[)(DPp><][-']?[:;=])|"
        r"<3|</3|:'\(|:\)|;\)|:D|:P|:O|xD|XD|:-\)|:-\(|:\(|T_T|>_<|\^_\^"
    )

    created: list[str] = []
    for c in text_cols:
        s = result[c].fillna("").astype(str)
        # Emoji counts
        result[f"{c}_emoji_count"] = s.apply(lambda t: len(emoji_pattern.findall(t)))
        result[f"{c}_emoticon_count"] = s.apply(lambda t: len(emoticon_pattern.findall(t)))
        result[f"{c}_emoji_total"] = result[f"{c}_emoji_count"] + result[f"{c}_emoticon_count"]
        char_len = s.str.len().replace(0, 1)
        result[f"{c}_emoji_ratio"] = (result[f"{c}_emoji_total"] / char_len).round(4)
        # List emojis found
        result[f"{c}_emojis_found"] = s.apply(
            lambda t: ",".join(emoji_pattern.findall(t)[:10])
        )
        created.extend([
            f"{c}_emoji_count", f"{c}_emoticon_count", f"{c}_emoji_total",
            f"{c}_emoji_ratio", f"{c}_emojis_found",
        ])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Extracted {len(created)} emoji features from {len(text_cols)} column(s)",
    )
