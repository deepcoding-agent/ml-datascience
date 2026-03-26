"""handle_cumulative_features handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_cumulative_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Add cumulative sum, mean, min, max as new feature columns."""
    col = params.get("column")
    col, err = BaseHandler.require_column(df, col, params.get("column", ""))
    if err:
        return err
    if col not in BaseHandler.get_numeric_cols(df):
        return HandlerResult(success=False, error=f"\'{col}\' is not numeric")
    result = df.copy()
    result[f"{col}_cumsum"] = result[col].cumsum()
    result[f"{col}_cummean"] = result[col].expanding().mean()
    result[f"{col}_cummax"] = result[col].cummax()
    result[f"{col}_cummin"] = result[col].cummin()
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Created cumulative features for \'{col}\': cumsum, cummean, cummax, cummin")
