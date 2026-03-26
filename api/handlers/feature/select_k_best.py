"""handle_select_k_best handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_select_k_best(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Select top K features using statistical tests (f_classif or f_regression)."""
    target = params.get("column")
    k = params.get("k", 10)
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not target or target not in df.columns:
        target = num_cols[-1] if num_cols else None
    if target is None:
        return HandlerResult(success=False, error="No target column found")

    try:
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        feature_cols = [c for c in num_cols if c != target]
        X = df[feature_cols].dropna()
        y = df.loc[X.index, target]

        scorer = f_classif if y.nunique() <= 10 else f_regression
        k_actual = min(k, len(feature_cols))
        selector = SelectKBest(scorer, k=k_actual)
        selector.fit(X, y)

        scores = pd.DataFrame({
            "feature": feature_cols,
            "score": selector.scores_,
            "selected": selector.get_support(),
        }).sort_values("score", ascending=True)

        fig = px.bar(scores, x="score", y="feature", orientation="h",
                     color="selected", color_discrete_map={True: "#FB8C3C", False: "#E0E0E0"})
        _style(fig, title=f"Top {k_actual} Features", showlegend=False)

        selected = scores[scores["selected"]]["feature"].tolist()
        result = df[selected + [target]]
        return HandlerResult(
            success=True, result_df=result, charts_plotly=[fig.to_json()],
            output_type="generate",
            summary=f"Selected top {k_actual} features (target='{target}'): {selected}",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"SelectKBest error: {e}")
