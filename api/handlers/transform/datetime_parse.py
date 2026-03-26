"""handle_datetime_parse handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_datetime_parse(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Parse string column to datetime type."""
    col = params.get("column")
    fmt = params.get("format")  # None = auto-detect
    col, err = BaseHandler.require_column(df, col, params.get("column", ""))
    if err:
        return err
    result = df.copy()
    try:
        if fmt:
            result[col] = pd.to_datetime(result[col], format=fmt, errors="coerce")
        else:
            result[col] = pd.to_datetime(result[col], format="mixed", errors="coerce")
        parsed = result[col].notna().sum()
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Parsed \'{col}\' to datetime ({parsed}/{len(df)} successful)")
    except Exception as e:
        return HandlerResult(success=False, error=f"Failed to parse dates: {e}")
