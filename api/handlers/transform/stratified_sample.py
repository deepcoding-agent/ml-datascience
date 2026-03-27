"""handle_stratified_sample handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_stratified_sample(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Stratified random sampling — preserves class ratios of a target column."""
    column = params.get("column") or params.get("target")
    fraction = float(params.get("fraction", 0.5))
    n = params.get("n")

    if not column or column not in df.columns:
        return HandlerResult(success=False, error="Specify 'column' (stratification column) for stratified sampling.")

    if fraction <= 0 or fraction > 1:
        fraction = 0.5

    try:
        groups = []
        for _, group in df.groupby(column):
            if n:
                n_per_group = max(int(int(n) / df[column].nunique()), 1)
                groups.append(group.sample(n=min(n_per_group, len(group)), random_state=42))
            else:
                groups.append(group.sample(frac=fraction, random_state=42))
        result = pd.concat(groups).reset_index(drop=True)
    except Exception as e:
        return HandlerResult(success=False, error=f"Stratified sampling failed: {e}")

    after_dist = dict(result[column].value_counts(normalize=True).round(3))

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Stratified sample on '{column}': {len(df)} → {len(result)} rows (fraction={fraction}). Class ratios preserved: {after_dist}",
    )
