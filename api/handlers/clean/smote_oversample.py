"""handle_smote_oversample handler."""
from __future__ import annotations

import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_smote_oversample(df: pd.DataFrame, params: dict) -> HandlerResult:
    """SMOTE oversampling — generate synthetic minority samples via k-NN interpolation."""
    from imblearn.over_sampling import SMOTE

    target_col = params.get("column") or params.get("target")
    k_neighbors = int(params.get("k_neighbors", 5))

    if not target_col or target_col not in df.columns:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_low = [c for c in df.select_dtypes(include="number").columns if df[c].nunique() <= 10]
        target_col = (num_low + cat_cols + [df.columns[-1]])[0]

    result = df.dropna(subset=[target_col]).copy()
    y = result[target_col]
    X = result.drop(columns=[target_col])
    num_cols = X.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return HandlerResult(success=False, error="SMOTE requires at least one numeric feature column.")

    X_num = X[num_cols].fillna(X[num_cols].median())

    min_class_count = int(y.value_counts().min())
    k = min(k_neighbors, max(min_class_count - 1, 1))

    try:
        sm = SMOTE(k_neighbors=k, random_state=42)
        X_res, y_res = sm.fit_resample(X_num, y)
    except Exception as e:
        return HandlerResult(success=False, error=f"SMOTE failed: {e}")

    out = pd.DataFrame(X_res, columns=num_cols)
    out[target_col] = y_res.values

    before = dict(y.value_counts())
    after = dict(y_res.value_counts())
    return HandlerResult(
        success=True, result_df=out, output_type="generate",
        summary=f"SMOTE oversampling on '{target_col}': {len(df)} → {len(out)} rows. Before: {before}, After: {after}",
    )
