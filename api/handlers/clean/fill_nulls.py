"""handle_fill_nulls handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fill_nulls(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    strategy = params.get("strategy", "auto")
    result = df.copy()
    filled: dict[str, str] = {}

    cols_to_fill = [col] if col and col in result.columns else result.columns.tolist()

    for c in cols_to_fill:
        if result[c].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(result[c]):
            if strategy == "auto":
                s = "median" if abs(result[c].skew()) > 1 else "mean"
            else:
                s = strategy
            if s == "median":
                result[c] = result[c].fillna(result[c].median())
            elif s == "mean":
                result[c] = result[c].fillna(result[c].mean())
            elif s == "zero":
                result[c] = result[c].fillna(0)
            else:
                result[c] = result[c].fillna(result[c].median())
            filled[c] = s
        else:
            mode = result[c].mode()
            result[c] = result[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
            filled[c] = "mode"

    remaining = int(result.isnull().sum().sum())
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Filled nulls in {len(filled)} columns. Remaining nulls: {remaining}",
                         metadata={"strategies": filled})
