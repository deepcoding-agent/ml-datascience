"""handle_add_column handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_add_column(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    expression = params.get("expression", "")
    if not col:
        return HandlerResult(success=False, error="No column name provided")
    result = df.copy()
    try:
        result[col] = result.eval(expression) if expression else 0
    except Exception as e:
        return HandlerResult(success=False, error=f"Expression error: {e}")
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Added column '{col}'")
