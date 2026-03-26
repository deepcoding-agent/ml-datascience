"""handle_string_slice handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_string_slice(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extract substring from a text column by character positions."""
    col = params.get("column")
    start = int(params.get("start", 0))
    end = params.get("end")
    end = int(end) if end is not None else None
    new_name = params.get("new_name")
    col, err = BaseHandler.require_column(df, col, params.get("column", ""))
    if err:
        return err
    result = df.copy()
    out_col = new_name or f"{col}_slice"
    result[out_col] = result[col].astype(str).str[start:end]
    desc = f"[{start}:{end}]" if end else f"[{start}:]"
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Extracted substring {desc} from \'{col}\' → \'{out_col}\'")
