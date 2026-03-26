"""handle_text_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_text_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract text statistics: length, word_count, digit_count, uppercase_ratio."""
    col = params.get("column")
    result = df.copy()
    str_cols = (
        [col] if col and col in result.columns
        else result.select_dtypes(include="object").columns.tolist()
    )
    if not str_cols:
        return HandlerResult(success=False, error="No string columns found")

    created = []
    for c in str_cols:
        s = result[c].fillna("").astype(str)
        result[f"{c}_len"] = s.str.len()
        result[f"{c}_word_count"] = s.str.split().str.len().fillna(0).astype(int)
        result[f"{c}_digit_count"] = s.str.count(r"\d")
        str_len = s.str.len().replace(0, 1)
        result[f"{c}_upper_ratio"] = (s.str.count(r"[A-Z]") / str_len).round(4)
        created.extend([f"{c}_len", f"{c}_word_count", f"{c}_digit_count", f"{c}_upper_ratio"])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Extracted {len(created)} text features from {len(str_cols)} column(s)",
    )
