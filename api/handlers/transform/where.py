"""handle_where handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_where(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Replace values where condition is NOT met (like np.where).
    Keeps values matching condition, replaces others."""
    col = params.get("column")
    operator = params.get("operator", ">")
    value = params.get("value")
    replacement = params.get("replacement", np.nan)
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    if value is None:
        return HandlerResult(success=False, error="Specify value= parameter")

    result = df.copy()
    ops = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
    method = ops.get(operator, "gt")
    try:
        if pd.api.types.is_numeric_dtype(result[col]):
            value = float(value)
        mask = getattr(result[col], method)(value)
        result[col] = result[col].where(mask, other=replacement)
    except Exception as e:
        return HandlerResult(success=False, error=f"Where error: {e}")

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Replaced '{col}' where NOT ({operator} {value}) → {replacement}",
    )
