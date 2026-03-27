"""handle_expanding_window handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_expanding_window(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create expanding (cumulative) window statistics for numeric columns."""
    column = params.get("column")
    stats_list = params.get("stats", ["mean", "std", "min", "max"])
    if isinstance(stats_list, str):
        stats_list = [s.strip() for s in stats_list.split(",")]

    result = df.copy()
    num_cols = [column] if column and column in result.select_dtypes(include="number").columns.tolist() \
        else result.select_dtypes(include="number").columns.tolist()

    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns found for expanding window.")

    created = []
    for col in num_cols:
        expanding = result[col].expanding(min_periods=1)
        for stat in stats_list:
            new_col = f"{col}_expanding_{stat}"
            if stat == "mean":
                result[new_col] = expanding.mean()
            elif stat == "std":
                result[new_col] = expanding.std()
            elif stat == "min":
                result[new_col] = expanding.min()
            elif stat == "max":
                result[new_col] = expanding.max()
            elif stat == "sum":
                result[new_col] = expanding.sum()
            elif stat == "median":
                result[new_col] = expanding.median()
            else:
                continue
            created.append(new_col)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} expanding window features for {len(num_cols)} columns: {created[:5]}{'...' if len(created) > 5 else ''}",
    )
