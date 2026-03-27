"""handle_correlation_select handler."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_correlation_select(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Drop one of each pair of highly correlated features (keeps first, drops second)."""
    threshold = float(params.get("threshold", 0.95))
    target_col = params.get("column") or params.get("target")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if target_col and target_col in num_cols:
        num_cols = [c for c in num_cols if c != target_col]

    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns for correlation-based selection.")

    corr = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    to_drop: list[str] = []
    pairs: list[tuple[str, str, float]] = []
    for col in upper.columns:
        high = upper.index[upper[col] > threshold].tolist()
        if high:
            to_drop.append(col)
            for h in high:
                pairs.append((h, col, round(float(corr.loc[h, col]), 4)))

    to_drop = list(set(to_drop))
    result = df.drop(columns=to_drop).copy()

    pair_df = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "correlation"]) if pairs else pd.DataFrame()

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Correlation selection (threshold={threshold}): dropped {len(to_drop)} features: {to_drop}. {len(pairs)} highly correlated pairs found.",
        metadata={"dropped": to_drop, "correlated_pairs": pairs},
    )
