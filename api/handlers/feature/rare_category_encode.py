"""handle_rare_category_encode handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_rare_category_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Group rare categories (< N occurrences) into 'Other'."""
    col = params.get("column")
    min_count = params.get("min_count", 5)
    result = df.copy()
    cat_cols = (
        [col] if col and col in result.columns
        else result.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    modified = []
    for c in cat_cols:
        counts = result[c].value_counts()
        rare = counts[counts < min_count].index
        if len(rare) > 0:
            result[c] = result[c].where(~result[c].isin(rare), "Other")
            modified.append(f"{c} ({len(rare)} rare)")
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Grouped rare categories (< {min_count}) into 'Other': {modified}",
    )
