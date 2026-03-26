"""handle_filter handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_filter(df: pd.DataFrame, params: dict) -> HandlerResult:
    col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
    if err:
        return err
    op = params.get("operator", "==")
    val = params.get("value")
    if val is None:
        return HandlerResult(success=False, error="No filter value provided")
    ops = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
    method = ops.get(op, "eq")
    try:
        if pd.api.types.is_numeric_dtype(df[col]):
            val = float(val)
        result = df[getattr(df[col], method)(val)]
    except Exception as e:
        return HandlerResult(success=False, error=f"Filter error: {e}")
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Filtered {col} {op} {val}: {len(df):,} → {len(result):,} rows")
