"""handle_leave_one_out_encode handler."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_leave_one_out_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Leave-one-out target encoding — reduces overfitting vs naive target encoding."""
    column = params.get("column")
    target_col = params.get("target")

    if not target_col or target_col not in df.columns:
        return HandlerResult(success=False, error="Specify 'target' param for leave-one-out encoding.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return HandlerResult(success=False, error=f"Target '{target_col}' must be numeric for LOO encoding.")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if column and column in cat_cols:
        cat_cols = [column]
    elif not cat_cols:
        return HandlerResult(success=False, error="No categorical columns found.")

    result = df.copy()
    encoded = []

    for col in cat_cols:
        if col == target_col:
            continue
        group_sum = result.groupby(col)[target_col].transform("sum")
        group_count = result.groupby(col)[target_col].transform("count")
        # LOO: (group_sum - this_row_value) / (group_count - 1)
        loo = (group_sum - result[target_col]) / (group_count - 1).replace(0, np.nan)
        new_col = f"{col}_loo_enc"
        result[new_col] = loo.fillna(result[target_col].mean())
        encoded.append(new_col)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Leave-one-out encoded {len(encoded)} columns: {encoded}",
    )
