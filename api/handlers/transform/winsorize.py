"""handle_winsorize handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_winsorize(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Cap extreme values at percentile bounds (e.g. 5th/95th)."""
    col = params.get("column")
    lower = float(params.get("lower", 0.05))
    upper = float(params.get("upper", 0.95))
    result = df.copy()

    if col and col in result.columns:
        cols = [col]
    else:
        cols = result.select_dtypes(include="number").columns.tolist()

    for c in cols:
        lo = result[c].quantile(lower)
        hi = result[c].quantile(upper)
        result[c] = result[c].clip(lower=lo, upper=hi)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Winsorized {len(cols)} column(s) at [{lower*100:.0f}th, {upper*100:.0f}th] percentile",
    )
