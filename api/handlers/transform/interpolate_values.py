"""handle_interpolate_values handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_interpolate_values(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Interpolate missing numeric values using linear interpolation."""
    col = params.get("column")
    method = params.get("method", "linear")
    result = df.copy()

    if col and col in result.columns:
        cols = [col]
    else:
        cols = result.select_dtypes(include="number").columns.tolist()

    if not cols:
        return HandlerResult(success=False, error="No numeric columns to interpolate")

    before = int(result[cols].isna().sum().sum())
    for c in cols:
        result[c] = result[c].interpolate(method=method)
    after = int(result[cols].isna().sum().sum())
    filled = before - after

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Interpolated ({method}) {len(cols)} column(s): {filled} nulls filled",
    )
