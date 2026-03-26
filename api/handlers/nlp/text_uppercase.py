"""handle_text_uppercase handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.nlp._helpers import _get_text_cols
from api.logger import get_logger
log = get_logger(__name__)

def handle_text_uppercase(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Convert text column to uppercase."""
    col = params.get("column")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")
    target = text_cols[0]
    result = df.copy()
    result[target] = result[target].astype(str).str.upper()
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Converted \'{target}\' to UPPERCASE")
