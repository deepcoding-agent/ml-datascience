"""handle_assign_value handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_assign_value(df: pd.DataFrame, params: dict) -> HandlerResult:
    col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
    if err:
        return err
    val = params.get("value")
    if val is None:
        return HandlerResult(success=False, error="No value to assign")
    result = df.copy()
    result[col] = val
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Set all {len(result):,} rows of '{col}' to {val}")
