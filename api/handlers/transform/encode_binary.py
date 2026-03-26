"""handle_encode_binary handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_encode_binary(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Encode a column as binary (0/1) based on a threshold or specific value."""
    col = params.get("column")
    threshold = params.get("threshold")
    positive_value = params.get("value")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")

    result = df.copy()
    if threshold is not None:
        result[f"{col}_binary"] = (result[col] >= float(threshold)).astype(int)
        desc = f"'{col}' >= {threshold} → 1, else 0"
    elif positive_value is not None:
        result[f"{col}_binary"] = (result[col] == positive_value).astype(int)
        desc = f"'{col}' == {positive_value} → 1, else 0"
    else:
        # Auto: if 2 unique values, encode the more common as 1
        uniques = result[col].dropna().unique()
        if len(uniques) == 2:
            result[f"{col}_binary"] = (result[col] == uniques[0]).astype(int)
            desc = f"'{col}': {uniques[0]}=1, {uniques[1]}=0"
        else:
            return HandlerResult(success=False, error="Specify threshold or value for binary encoding")

    ones = int(result[f"{col}_binary"].sum())
    zeros = len(result) - ones
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Binary encoded: {desc} ({ones} positive, {zeros} negative)",
    )
