"""handle_covariance handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_covariance(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compute covariance matrix for numeric columns."""
    nums = BaseHandler.get_numeric_cols(df)
    if len(nums) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")
    cov = df[nums].cov().round(4)
    result = cov.reset_index().rename(columns={"index": "column"})
    return HandlerResult(success=True, result_df=result, output_type="query",
                         summary=f"Covariance matrix for {len(nums)} numeric columns")
