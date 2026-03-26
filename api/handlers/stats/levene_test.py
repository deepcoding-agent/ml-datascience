"""handle_levene_test handler."""
from __future__ import annotations
import pandas as pd
from scipy.stats import levene
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_levene_test(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Levene's test for equality of variances across groups."""
    col = params.get("column")
    group_col = params.get("group")
    nums = BaseHandler.get_numeric_cols(df)
    cats = BaseHandler.get_categorical_cols(df)
    if not col:
        col = nums[0] if nums else None
    if not group_col:
        group_col = cats[0] if cats else None
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Numeric column required. Available: {nums}")
    if not group_col or group_col not in df.columns:
        return HandlerResult(success=False, error=f"Group column required. Available: {cats}")
    groups = [g[col].dropna().values for _, g in df.groupby(group_col) if len(g[col].dropna()) >= 2]
    if len(groups) < 2:
        return HandlerResult(success=False, error="Need at least 2 groups with data")
    stat, p = levene(*groups)
    sig = "significant" if p < 0.05 else "not significant"
    rows = [{"test": "Levene", "column": col, "group_by": group_col,
             "statistic": round(stat, 4), "p_value": round(p, 6),
             "result": sig, "n_groups": len(groups)}]
    return HandlerResult(success=True, result_df=pd.DataFrame(rows), output_type="query",
                         summary=f"Levene's test: variances are {sig} (p={p:.4f})")
