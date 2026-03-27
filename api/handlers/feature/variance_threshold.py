"""handle_variance_threshold handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_variance_threshold(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Drop features with variance below threshold — removes near-constant columns."""
    from sklearn.feature_selection import VarianceThreshold

    threshold = float(params.get("threshold", 0.01))

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns found.")

    X_num = df[num_cols].fillna(df[num_cols].median())

    try:
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_num)
    except Exception as e:
        return HandlerResult(success=False, error=f"Variance threshold failed: {e}")

    mask = selector.get_support()
    selected = [c for c, s in zip(num_cols, mask) if s]
    dropped = [c for c, s in zip(num_cols, mask) if not s]
    variances = {c: round(float(v), 6) for c, v in zip(num_cols, selector.variances_)}

    non_num_cols = [c for c in df.columns if c not in num_cols]
    result = df[non_num_cols + selected].copy()

    score_df = pd.DataFrame({
        "feature": num_cols,
        "variance": [round(float(v), 6) for v in selector.variances_],
        "selected": ["Yes" if s else "No" for s in mask],
    }).sort_values("variance", ascending=False)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Variance threshold ({threshold}): kept {len(selected)}/{len(num_cols)} numeric features. Dropped: {dropped}",
        metadata={"selected_features": selected, "dropped": dropped, "variances": variances},
    )
