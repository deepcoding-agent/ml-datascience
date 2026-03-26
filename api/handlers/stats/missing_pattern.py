"""handle_missing_pattern handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_missing_pattern(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Analyze which columns tend to be missing together."""
    null_mask = df.isnull()
    null_cols = [c for c in df.columns if null_mask[c].any()]
    if not null_cols:
        return HandlerResult(success=True, result_df=pd.DataFrame([{"pattern": "No missing values"}]),
                             summary="No missing values found")
    pattern = null_mask[null_cols].value_counts().head(10).reset_index()
    pattern.columns = list(null_cols) + ["count"]
    pattern["pct"] = (pattern["count"] / len(df) * 100).round(2)
    # Co-occurrence: which column pairs are missing together
    co = null_mask[null_cols].T.dot(null_mask[null_cols])
    return HandlerResult(success=True, result_df=pattern,
                         summary=f"Missing patterns across {len(null_cols)} columns (top {len(pattern)} patterns)")
