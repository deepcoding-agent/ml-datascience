"""handle_clip_outliers handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_clip_outliers(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Clip outliers using IQR or z-score method."""
    method = params.get("method", "iqr")
    col = params.get("column")
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
    clipped_info: dict[str, int] = {}

    for c in cols:
        before_outliers = 0
        if method == "zscore":
            mean, std = result[c].mean(), result[c].std()
            if std == 0:
                continue
            z = (result[c] - mean) / std
            mask = z.abs() > 3
            before_outliers = int(mask.sum())
            result.loc[mask & (z > 0), c] = mean + 3 * std
            result.loc[mask & (z < 0), c] = mean - 3 * std
        else:  # iqr
            q1 = result[c].quantile(0.25)
            q3 = result[c].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (result[c] < lower) | (result[c] > upper)
            before_outliers = int(mask.sum())
            result[c] = result[c].clip(lower=lower, upper=upper)
        if before_outliers > 0:
            clipped_info[c] = before_outliers

    total = sum(clipped_info.values())
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Clipped {total:,} outliers ({method.upper()}) across {len(clipped_info)} columns",
        metadata={"clipped": clipped_info},
    )
