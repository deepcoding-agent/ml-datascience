"""handle_mutual_info_select handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_mutual_info_select(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Rank and select features by mutual information with target, then drop low-scoring ones."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    target_col = params.get("column") or params.get("target")
    threshold = float(params.get("threshold", 0.01))
    top_k = params.get("top_k")

    if not target_col or target_col not in df.columns:
        return HandlerResult(success=False, error="Specify target column via 'column' param for mutual info selection.")

    work = df.dropna(subset=[target_col]).copy()
    y = work[target_col]
    X = work.drop(columns=[target_col])
    num_cols = X.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return HandlerResult(success=False, error="No numeric feature columns found.")

    X_num = X[num_cols].fillna(X[num_cols].median())

    is_clf = y.nunique() <= 20 and (y.dtype in ("object", "category", "bool") or y.nunique() <= 10)
    mi_func = mutual_info_classif if is_clf else mutual_info_regression

    try:
        mi_scores = mi_func(X_num, y, random_state=42)
    except Exception as e:
        return HandlerResult(success=False, error=f"Mutual information calculation failed: {e}")

    score_df = pd.DataFrame({"feature": num_cols, "mi_score": mi_scores}).sort_values("mi_score", ascending=False)

    if top_k:
        selected = score_df.head(int(top_k))["feature"].tolist()
    else:
        selected = score_df[score_df["mi_score"] >= threshold]["feature"].tolist()

    if not selected:
        selected = score_df.head(max(len(num_cols) // 2, 1))["feature"].tolist()

    score_df["selected"] = score_df["feature"].isin(selected).map({True: "Yes", False: "No"})

    fig = px.bar(score_df, x="mi_score", y="feature", orientation="h", color="selected",
                 color_discrete_map={"Yes": "#E8500A", "No": "#C0C0C0"})
    _style(fig, title=f"Mutual Information Scores vs '{target_col}'")

    kept = work[selected + [target_col]]
    return HandlerResult(
        success=True, result_df=kept, output_type="generate",
        charts_plotly=[fig.to_json()],
        summary=f"Mutual info selection: kept {len(selected)}/{len(num_cols)} features (threshold={threshold}). Selected: {selected}",
        metadata={"selected_features": selected, "scores": dict(zip(score_df["feature"], score_df["mi_score"].round(4)))},
    )
