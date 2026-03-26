"""handle_feature_importance handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_feature_importance(df: pd.DataFrame, params: dict) -> HandlerResult:
    target = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not target or target not in df.columns:
        target = num_cols[-1] if num_cols else None
    if target is None:
        return HandlerResult(success=False, error="No target column specified or found")
    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        X = df[num_cols].drop(columns=[target], errors="ignore").dropna()
        y = df.loc[X.index, target]
        if y.nunique() <= 10:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        imp = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        imp = imp.sort_values("importance", ascending=True)
        fig = px.bar(imp, x="importance", y="feature", orientation="h")
        fig.update_traces(marker_color="#FB8C3C")
        _style(fig, title="Feature Importance")
        return HandlerResult(success=True, result_df=imp, charts_plotly=[fig.to_json()],
                             summary=f"Feature importance (target='{target}', model=RandomForest)")
    except Exception as e:
        return HandlerResult(success=False, error=f"Feature importance error: {e}")
