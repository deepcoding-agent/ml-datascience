"""handle_round_to_nearest handler."""
from __future__ import annotations
import numpy as np
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_round_to_nearest(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Round numeric column to nearest N (e.g. nearest 5, 10, 100)."""
    col = params.get("column")
    nearest = float(params.get("nearest", params.get("value", 10)))
    col, err = BaseHandler.require_column(df, col, params.get("column", ""))
    if err:
        return err
    if col not in BaseHandler.get_numeric_cols(df):
        return HandlerResult(success=False, error=f"\'{col}\' is not numeric")
    result = df.copy()
    result[col] = (np.round(result[col] / nearest) * nearest)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Rounded \'{col}\' to nearest {nearest}")
