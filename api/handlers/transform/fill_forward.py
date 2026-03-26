"""handle_fill_forward handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_fill_forward(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Forward-fill (ffill) null values only."""
    col = params.get("column")
    result = df.copy()

    if col and col in result.columns:
        before = int(result[col].isna().sum())
        result[col] = result[col].ffill()
        after = int(result[col].isna().sum())
        filled = before - after
        desc = f"Forward-filled '{col}': {filled} nulls filled"
    else:
        before = int(result.isna().sum().sum())
        result = result.ffill()
        after = int(result.isna().sum().sum())
        filled = before - after
        desc = f"Forward-filled all columns: {filled} nulls filled"

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=desc,
    )
