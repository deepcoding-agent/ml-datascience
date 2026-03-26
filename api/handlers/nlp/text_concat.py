"""handle_text_concat handler."""
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


def handle_text_concat(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Combine multiple text columns into a single corpus column.
    Useful before vectorization when text is split across multiple fields."""
    columns = params.get("columns")
    separator = params.get("separator", " ")
    new_name = params.get("new_name", "text_combined")
    result = df.copy()

    if columns and isinstance(columns, list):
        text_cols = [c for c in columns if c in result.columns]
    else:
        text_cols = result.select_dtypes(include="object").columns.tolist()

    if len(text_cols) < 2:
        return HandlerResult(
            success=False,
            error=f"Need at least 2 text columns to concatenate. Found: {text_cols}",
        )

    result[new_name] = result[text_cols].fillna("").astype(str).agg(separator.join, axis=1)
    result[new_name] = result[new_name].str.strip()

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Concatenated {len(text_cols)} columns → '{new_name}': {', '.join(text_cols)}",
    )
