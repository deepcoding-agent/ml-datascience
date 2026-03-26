"""handle_confidence_interval handler."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_confidence_interval(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compute confidence intervals for numeric columns."""
    confidence = float(params.get("confidence", 0.95))
    col = params.get("column")
    nums = [col] if col and col in df.columns else BaseHandler.get_numeric_cols(df)
    if not nums:
        return HandlerResult(success=False, error="No numeric columns found")
    rows = []
    for c in nums:
        data = df[c].dropna()
        if len(data) < 2:
            continue
        mean = data.mean()
        se = sp_stats.sem(data)
        ci = sp_stats.t.interval(confidence, len(data)-1, loc=mean, scale=se)
        rows.append({"column": c, "mean": round(mean, 4), "ci_lower": round(ci[0], 4),
                      "ci_upper": round(ci[1], 4), "confidence": confidence, "n": len(data)})
    result = pd.DataFrame(rows)
    return HandlerResult(success=True, result_df=result, output_type="query",
                         summary=f"{confidence*100:.0f}% confidence intervals for {len(rows)} columns")
