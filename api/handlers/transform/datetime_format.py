"""handle_datetime_format handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_datetime_format(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Format datetime column to string with specified format."""
    col = params.get("column")
    fmt = params.get("format", "%Y-%m-%d")
    new_name = params.get("new_name")
    col, err = BaseHandler.require_column(df, col, params.get("column", ""))
    if err:
        return err
    result = df.copy()
    dt = pd.to_datetime(result[col], format="mixed", errors="coerce")
    out_col = new_name or f"{col}_formatted"
    result[out_col] = dt.dt.strftime(fmt).fillna("")
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Formatted \'{col}\' → \'{out_col}\' with format \'{fmt}\'")
