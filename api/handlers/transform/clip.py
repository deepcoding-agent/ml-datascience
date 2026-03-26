"""handle_clip handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_clip(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Clip numeric values to min/max bounds."""
    col = params.get("column")
    lower = params.get("min")
    upper = params.get("max")
    result = df.copy()

    if col and col in result.columns:
        cols = [col]
    else:
        cols = result.select_dtypes(include="number").columns.tolist()

    for c in cols:
        lo = float(lower) if lower is not None else None
        hi = float(upper) if upper is not None else None
        result[c] = result[c].clip(lower=lo, upper=hi)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Clipped {len(cols)} column(s) to [{lower}, {upper}]",
    )
