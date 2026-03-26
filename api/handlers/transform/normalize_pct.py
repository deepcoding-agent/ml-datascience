"""handle_normalize_pct handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_normalize_pct(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Normalize numeric columns to percentages (row-wise or column-wise)."""
    axis = params.get("axis", "columns")  # columns (per-row) or index (per-column)
    result = df.copy()
    num_cols = result.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns")

    if axis == "columns":
        # Each row sums to 100%
        row_sums = result[num_cols].sum(axis=1).replace(0, 1)
        for c in num_cols:
            result[c] = (result[c] / row_sums * 100).round(2)
        desc = "row-wise (each row sums to 100%)"
    else:
        # Each column sums to 100%
        for c in num_cols:
            total = result[c].sum()
            if total != 0:
                result[c] = (result[c] / total * 100).round(2)
        desc = "column-wise (each column sums to 100%)"

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Normalized {len(num_cols)} columns to percentages ({desc})",
    )
