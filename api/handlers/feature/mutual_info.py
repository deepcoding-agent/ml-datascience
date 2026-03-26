"""handle_mutual_info handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_mutual_info(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Mutual information scores — works for both numeric and categorical features."""
    target = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not target or target not in df.columns:
        target = num_cols[-1] if num_cols else None
    if target is None:
        return HandlerResult(success=False, error="No target column found")

    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        feature_cols = [c for c in num_cols if c != target]
        clean_df = df[feature_cols + [target]].dropna()
        X = clean_df[feature_cols]
        y = clean_df[target]

        scorer = mutual_info_classif if y.nunique() <= 10 else mutual_info_regression
        mi_scores = scorer(X, y, random_state=42)

        result = pd.DataFrame({
            "feature": feature_cols,
            "mutual_info": np.round(mi_scores, 4),
        }).sort_values("mutual_info", ascending=True).reset_index(drop=True)

        fig = px.bar(result, x="mutual_info", y="feature", orientation="h")
        fig.update_traces(marker_color="#FB8C3C")
        _style(fig, title=f"Mutual Information (target: {target})")

        return HandlerResult(
            success=True, result_df=result, charts_plotly=[fig.to_json()],
            summary=f"Mutual information for {len(feature_cols)} features (target='{target}')",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Mutual info error: {e}")
