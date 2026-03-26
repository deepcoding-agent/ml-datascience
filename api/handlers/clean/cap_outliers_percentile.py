"""handle_cap_outliers_percentile handler."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_cap_outliers_percentile(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Cap outliers at Nth percentile (e.g. 1st and 99th)."""
    col = params.get("column")
    lower_pct = params.get("lower", 1)
    upper_pct = params.get("upper", 99)
    result = df.copy()
    cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
    capped: dict[str, int] = {}

    for c in cols:
        lo = result[c].quantile(lower_pct / 100)
        hi = result[c].quantile(upper_pct / 100)
        mask = (result[c] < lo) | (result[c] > hi)
        count = int(mask.sum())
        if count > 0:
            result[c] = result[c].clip(lower=lo, upper=hi)
            capped[c] = count

    total = sum(capped.values())
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Capped {total:,} values at p{lower_pct}/p{upper_pct} across {len(capped)} column(s)",
        metadata={"capped": capped},
    )
