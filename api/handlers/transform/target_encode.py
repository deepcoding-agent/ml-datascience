"""handle_target_encode_transform handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_target_encode_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Target encoding — replace each category with the mean of the target variable (with smoothing)."""
    column = params.get("column")
    target_col = params.get("target")
    smoothing = float(params.get("smoothing", 10))

    if not target_col or target_col not in df.columns:
        return HandlerResult(success=False, error="Specify 'target' param for target encoding.")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if column and column in cat_cols:
        cat_cols = [column]
    elif column:
        return HandlerResult(success=False, error=f"Column '{column}' is not categorical.")

    if not cat_cols:
        return HandlerResult(success=False, error="No categorical columns found.")

    result = df.copy()
    global_mean = float(result[target_col].mean())
    encoded = []

    for col in cat_cols:
        if col == target_col:
            continue
        agg = result.groupby(col)[target_col].agg(["mean", "count"])
        # Bayesian smoothing: blend category mean with global mean
        smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
        mapping = smooth.to_dict()
        new_col = f"{col}_target_enc"
        result[new_col] = result[col].map(mapping).fillna(global_mean)
        encoded.append(new_col)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Target-encoded {len(encoded)} columns (smoothing={smoothing}): {encoded}",
    )
