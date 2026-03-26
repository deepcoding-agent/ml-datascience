"""handle_target_binary_encode handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_target_binary_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Encode a numeric target as binary (1 if above median, 0 otherwise)."""
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not col or col not in df.columns:
        col = num_cols[-1] if num_cols else None
    if col is None:
        return HandlerResult(success=False, error="No target column found")
    result = df.copy()
    median_val = result[col].median()
    result[f"{col}_binary"] = (result[col] >= median_val).astype(int)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Binary-encoded '{col}' (1 if >= median {median_val:.2f}, else 0)",
    )
