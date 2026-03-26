"""handle_text_pattern_match handler."""
from __future__ import annotations
import re
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.nlp._helpers import _get_text_cols
from api.logger import get_logger
log = get_logger(__name__)

def handle_text_pattern_match(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Filter rows where text column matches a regex pattern."""
    col = params.get("column")
    pattern = params.get("pattern", params.get("value", ""))
    if not pattern:
        return HandlerResult(success=False, error="Provide a 'pattern' (regex) to match")
    text_cols = _get_text_cols(df, col)
    if not text_cols:
        return HandlerResult(success=False, error="No text columns found")
    target = text_cols[0]
    mask = df[target].astype(str).str.contains(pattern, flags=re.IGNORECASE, na=False)
    result = df[mask].reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Filtered \'{target}\' by pattern \'{pattern}\': {mask.sum()}/{len(df)} rows matched")
