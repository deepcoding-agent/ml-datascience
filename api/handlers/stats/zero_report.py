"""handle_zero_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_zero_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Count zero values per column (important for sparse data)."""
    rows = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            zero_count = int((df[c] == 0).sum())
            zero_pct = round(zero_count / len(df) * 100, 2)
        else:
            zero_count = int((df[c].isin(["0", "", " "])).sum())
            zero_pct = round(zero_count / len(df) * 100, 2)
        rows.append({"column": c, "zero_count": zero_count, "zero_pct": zero_pct})

    result = pd.DataFrame(rows).sort_values("zero_count", ascending=False).reset_index(drop=True)
    total = result["zero_count"].sum()
    return HandlerResult(success=True, result_df=result, summary=f"Total zeros/empty: {total:,}")
