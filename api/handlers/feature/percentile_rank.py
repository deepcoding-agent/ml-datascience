"""handle_percentile_rank handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_percentile_rank(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Add percentile rank (0-100) for numeric column(s)."""
    col = params.get("column")
    nums = [col] if col and col in df.columns else BaseHandler.get_numeric_cols(df)
    if not nums:
        return HandlerResult(success=False, error="No numeric columns found")
    result = df.copy()
    for c in nums:
        result[f"{c}_pct_rank"] = result[c].rank(pct=True).mul(100).round(2)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Added percentile rank for {len(nums)} columns")
