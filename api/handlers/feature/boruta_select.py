"""handle_boruta_select handler."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_boruta_select(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Boruta all-relevant feature selection using shadow features + random forest."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    target_col = params.get("column") or params.get("target")
    max_iter = int(params.get("max_iter", 50))

    if not target_col or target_col not in df.columns:
        return HandlerResult(success=False, error="Specify target column via 'column' param for Boruta selection.")

    work = df.dropna(subset=[target_col]).copy()
    y = work[target_col]
    X = work.drop(columns=[target_col])
    num_cols = X.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Boruta needs at least 2 numeric feature columns.")

    X_num = X[num_cols].fillna(X[num_cols].median())

    is_clf = y.nunique() <= 20 and (y.dtype in ("object", "category", "bool") or y.nunique() <= 10)

    try:
        from boruta import BorutaPy

        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=5) if is_clf \
            else RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=5)

        boruta = BorutaPy(estimator, n_estimators="auto", max_iter=max_iter, random_state=42, verbose=0)
        boruta.fit(X_num.values, y.values)
    except ImportError:
        return HandlerResult(success=False, error="boruta package not installed. Run: pip install boruta")
    except Exception as e:
        return HandlerResult(success=False, error=f"Boruta selection failed: {e}")

    confirmed = [c for c, s in zip(num_cols, boruta.support_) if s]
    tentative = [c for c, s in zip(num_cols, boruta.support_weak_) if s]
    rejected = [c for c in num_cols if c not in confirmed and c not in tentative]

    result_df = pd.DataFrame({
        "feature": num_cols,
        "ranking": [int(r) for r in boruta.ranking_],
        "status": ["Confirmed" if s else ("Tentative" if w else "Rejected")
                    for s, w in zip(boruta.support_, boruta.support_weak_)],
    }).sort_values("ranking")

    selected = confirmed + tentative if not confirmed else confirmed
    kept = work[selected + [target_col]] if selected else work

    return HandlerResult(
        success=True, result_df=kept, output_type="generate",
        summary=f"Boruta selection: {len(confirmed)} confirmed, {len(tentative)} tentative, {len(rejected)} rejected. Selected: {selected}",
        metadata={"confirmed": confirmed, "tentative": tentative, "rejected": rejected},
    )
