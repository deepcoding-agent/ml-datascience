"""handle_remove_outliers handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_remove_outliers(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Remove rows containing outlier values (IQR or z-score)."""
    method = params.get("method", "iqr")
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
    mask = pd.Series(True, index=result.index)

    for c in cols:
        if method == "zscore":
            mean, std = result[c].mean(), result[c].std()
            if std == 0:
                continue
            z = ((result[c] - mean) / std).abs()
            mask &= z <= 3
        else:  # iqr
            q1 = result[c].quantile(0.25)
            q3 = result[c].quantile(0.75)
            iqr = q3 - q1
            mask &= (result[c] >= q1 - 1.5 * iqr) & (result[c] <= q3 + 1.5 * iqr)

    before = len(result)
    result = result[mask].reset_index(drop=True)
    removed = before - len(result)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Removed {removed:,} outlier rows ({method.upper()}): {before:,} → {len(result):,}",
    )
