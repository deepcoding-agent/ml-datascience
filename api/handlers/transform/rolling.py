"""handle_rolling handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_rolling(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Rolling/moving window: mean, sum, std, min, max."""
    col = params.get("column")
    window = params.get("window", 3)
    func = params.get("agg", "mean")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
    if not col:
        return HandlerResult(success=False, error="No column for rolling")
    result = df.copy()
    roller = result[col].rolling(window=window, min_periods=1)
    result[f"{col}_rolling_{func}_{window}"] = getattr(roller, func)().round(4)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Rolling {func} (window={window}) of '{col}'")
