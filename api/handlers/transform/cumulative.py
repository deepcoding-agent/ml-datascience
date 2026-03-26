"""handle_cumulative handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_cumulative(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Cumulative sum, count, max, or min."""
    col = params.get("column")
    func = params.get("agg", "sum")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
    if not col:
        return HandlerResult(success=False, error="No column for cumulative")
    result = df.copy()
    if func == "sum":
        result[f"{col}_cumsum"] = result[col].cumsum()
    elif func == "max":
        result[f"{col}_cummax"] = result[col].cummax()
    elif func == "min":
        result[f"{col}_cummin"] = result[col].cummin()
    elif func == "count":
        result[f"{col}_cumcount"] = range(1, len(result) + 1)
    else:
        result[f"{col}_cumsum"] = result[col].cumsum()
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Cumulative {func} of '{col}'")
