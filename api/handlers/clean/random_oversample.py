"""handle_random_oversample handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_random_oversample(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Randomly duplicate minority class rows to balance class distribution."""
    from imblearn.over_sampling import RandomOverSampler

    target_col = params.get("column") or params.get("target")

    if not target_col or target_col not in df.columns:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_low = [c for c in df.select_dtypes(include="number").columns if df[c].nunique() <= 10]
        target_col = (num_low + cat_cols + [df.columns[-1]])[0]

    result = df.dropna(subset=[target_col]).copy()
    y = result[target_col]
    X = result.drop(columns=[target_col])

    try:
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
    except Exception as e:
        return HandlerResult(success=False, error=f"Random oversampling failed: {e}")

    out = pd.DataFrame(X_res, columns=X.columns)
    out[target_col] = y_res.values

    before = dict(y.value_counts())
    after = dict(y_res.value_counts())
    return HandlerResult(
        success=True, result_df=out, output_type="generate",
        summary=f"Random oversampling on '{target_col}': {len(df)} → {len(out)} rows. Before: {before}, After: {after}",
    )
