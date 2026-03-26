"""handle_granger_causality handler."""
from __future__ import annotations
import pandas as pd
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_granger_causality(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Granger causality test between two time-ordered numeric columns."""
    from statsmodels.tsa.stattools import grangercausalitytests
    col_x = params.get("column") or params.get("x")
    col_y = params.get("y") or params.get("target")
    max_lag = int(params.get("max_lag", 4))
    nums = BaseHandler.get_numeric_cols(df)
    if not col_x and len(nums) >= 2:
        col_x = nums[0]
    if not col_y and len(nums) >= 2:
        col_y = nums[1]
    if not col_x or not col_y:
        return HandlerResult(success=False, error="Need 2 numeric columns (x, y)")
    data = df[[col_y, col_x]].dropna()
    if len(data) < max_lag * 3:
        return HandlerResult(success=False, error=f"Need more data points (have {len(data)}, need {max_lag*3}+)")
    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    except Exception as e:
        return HandlerResult(success=False, error=f"Granger test failed: {e}")
    rows = []
    for lag in range(1, max_lag + 1):
        f_test = results[lag][0]["ssr_ftest"]
        rows.append({"lag": lag, "f_statistic": round(f_test[0], 4),
                      "p_value": round(f_test[1], 6),
                      "significant": "yes" if f_test[1] < 0.05 else "no"})
    return HandlerResult(success=True, result_df=pd.DataFrame(rows), output_type="query",
                         summary=f"Granger causality: does \'{col_x}\' cause \'{col_y}\'? Tested lags 1-{max_lag}")
