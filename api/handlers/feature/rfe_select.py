"""handle_rfe_select handler."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_rfe_select(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Recursive Feature Elimination — iteratively remove weakest features."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import RFE

    target_col = params.get("column") or params.get("target")
    n_features = params.get("n_features", None)

    if not target_col or target_col not in df.columns:
        return HandlerResult(success=False, error="Specify target column via 'column' param for RFE feature selection.")

    work = df.dropna(subset=[target_col]).copy()
    y = work[target_col]
    X = work.drop(columns=[target_col])
    num_cols = X.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="RFE needs at least 2 numeric feature columns.")

    X_num = X[num_cols].fillna(X[num_cols].median())
    n_select = int(n_features) if n_features else max(len(num_cols) // 2, 1)

    is_clf = y.nunique() <= 20 and y.dtype in ("object", "category", "bool") or y.nunique() <= 10
    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) if is_clf \
        else RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

    try:
        rfe = RFE(estimator, n_features_to_select=n_select, step=1)
        rfe.fit(X_num, y)
    except Exception as e:
        return HandlerResult(success=False, error=f"RFE failed: {e}")

    selected = [c for c, s in zip(num_cols, rfe.support_) if s]
    rankings = {c: int(r) for c, r in zip(num_cols, rfe.ranking_)}
    result_df = pd.DataFrame({
        "feature": num_cols,
        "ranking": [int(r) for r in rfe.ranking_],
        "selected": ["Yes" if s else "No" for s in rfe.support_],
    }).sort_values("ranking")

    kept = work[selected + [target_col]]
    return HandlerResult(
        success=True, result_df=kept, output_type="generate",
        summary=f"RFE selected {len(selected)}/{len(num_cols)} features: {selected}. Dropped: {[c for c in num_cols if c not in selected]}",
        metadata={"selected_features": selected, "rankings": rankings},
    )
