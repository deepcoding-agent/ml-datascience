"""handle_trim_text_length handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_trim_text_length(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Truncate text column values to a maximum character length."""
    col = params.get("column")
    max_len = int(params.get("max_length", params.get("value", 100)))
    col, err = BaseHandler.require_column(df, col, params.get("column", ""))
    if err:
        return err
    result = df.copy()
    before_avg = result[col].astype(str).str.len().mean()
    result[col] = result[col].astype(str).str[:max_len]
    after_avg = result[col].str.len().mean()
    trimmed = (before_avg > after_avg)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Trimmed \'{col}\' to max {max_len} chars (avg length: {before_avg:.0f} → {after_avg:.0f})")
