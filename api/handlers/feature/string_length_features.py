"""handle_string_length_features handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_string_length_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create length-based features from string columns (char_count, word_count)."""
    col = params.get("column")
    cats = BaseHandler.get_categorical_cols(df)
    cols = [col] if col and col in df.columns else cats
    if not cols:
        return HandlerResult(success=False, error="No text/string columns found")
    result = df.copy()
    created = []
    for c in cols[:5]:
        s = result[c].astype(str)
        result[f"{c}_char_len"] = s.str.len()
        result[f"{c}_word_count"] = s.str.split().str.len()
        created.extend([f"{c}_char_len", f"{c}_word_count"])
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Created string length features: {created}")
