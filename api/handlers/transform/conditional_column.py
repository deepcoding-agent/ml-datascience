"""handle_conditional_column handler."""
from __future__ import annotations
import numpy as np
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_conditional_column(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create new column with if/else logic based on a condition."""
    col = params.get("column")
    operator = params.get("operator", ">")
    value = params.get("value")
    true_val = params.get("true_value", params.get("true", 1))
    false_val = params.get("false_value", params.get("false", 0))
    new_name = params.get("new_name", f"{col}_flag")
    col, err = BaseHandler.require_column(df, col, params.get("column", ""))
    if err:
        return err
    result = df.copy()
    try:
        v = float(value) if value is not None else df[col].median()
    except (ValueError, TypeError):
        v = value
    ops = {">": "gt", ">=": "ge", "<": "lt", "<=": "le", "==": "eq", "!=": "ne"}
    op = ops.get(operator, "gt")
    mask = getattr(result[col], op)(v)
    result[new_name] = np.where(mask, true_val, false_val)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Created \'{new_name}\': {col} {operator} {v} → {true_val}/{false_val}")
