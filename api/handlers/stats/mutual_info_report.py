"""handle_mutual_info_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_mutual_info_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Mutual information scores between all numeric features."""
    from sklearn.feature_selection import mutual_info_regression

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")

    target_col = params.get("column", num_cols[-1])
    if target_col not in num_cols:
        target_col = num_cols[-1]
    feature_cols = [c for c in num_cols if c != target_col]
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    mi = mutual_info_regression(X, y, random_state=42)
    result = pd.DataFrame({"feature": feature_cols, "mutual_info": np.round(mi, 4)})
    result = result.sort_values("mutual_info", ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result,
                         summary=f"Mutual information scores vs '{target_col}'")
