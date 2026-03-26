"""handle_round_values handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_round_values(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Round numeric columns to N decimal places."""
    col = params.get("column")
    decimals = params.get("decimals", 2)
    result = df.copy()
    if col and col in result.columns:
        result[col] = result[col].round(decimals)
        summary = f"Rounded '{col}' to {decimals} decimals"
    else:
        num_cols = result.select_dtypes(include="number").columns.tolist()
        result[num_cols] = result[num_cols].round(decimals)
        summary = f"Rounded {len(num_cols)} numeric columns to {decimals} decimals"
    return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)
