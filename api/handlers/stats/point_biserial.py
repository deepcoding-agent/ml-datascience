"""handle_point_biserial handler."""
from __future__ import annotations
import pandas as pd
from scipy.stats import pointbiserialr
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_point_biserial(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Point-biserial correlation between binary and numeric columns."""
    nums = BaseHandler.get_numeric_cols(df)
    binary_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]
    if not binary_cols:
        return HandlerResult(success=False, error="No binary columns found")
    if not nums:
        return HandlerResult(success=False, error="No numeric columns found")
    rows = []
    for bc in binary_cols:
        vals = df[bc].dropna().unique()
        mapping = {vals[0]: 0, vals[1]: 1}
        binary_num = df[bc].map(mapping)
        for nc in nums:
            if nc == bc:
                continue
            mask = binary_num.notna() & df[nc].notna()
            if mask.sum() < 3:
                continue
            r, p = pointbiserialr(binary_num[mask], df[nc][mask])
            rows.append({"binary_col": bc, "numeric_col": nc,
                         "correlation": round(r, 4), "p_value": round(p, 6)})
    rows.sort(key=lambda r: abs(r["correlation"]), reverse=True)
    return HandlerResult(success=True, result_df=pd.DataFrame(rows), output_type="query",
                         summary=f"Point-biserial correlations: {len(rows)} pairs computed")
