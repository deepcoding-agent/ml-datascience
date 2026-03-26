"""handle_shape handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_shape(df: pd.DataFrame, params: dict) -> HandlerResult:
    mem = df.memory_usage(deep=True).sum() / 1024**2
    num_cols = len(df.select_dtypes(include="number").columns)
    cat_cols = len(df.select_dtypes(include=["object", "category"]).columns)
    null_total = int(df.isna().sum().sum())
    dup_count = int(df.duplicated().sum())
    result = pd.DataFrame([{
        "rows": df.shape[0], "columns": df.shape[1],
        "numeric_cols": num_cols, "categorical_cols": cat_cols,
        "total_nulls": null_total, "duplicates": dup_count,
        "memory_mb": round(mem, 2),
    }])
    summary = (
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns ({mem:.2f} MB) | "
        f"Numeric: {num_cols} | Categorical: {cat_cols} | "
        f"Nulls: {null_total:,} | Duplicates: {dup_count:,}"
    )
    return HandlerResult(success=True, result_df=result, summary=summary)
