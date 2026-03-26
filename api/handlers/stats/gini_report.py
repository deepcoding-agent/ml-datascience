"""handle_gini_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_gini_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Gini impurity per categorical column (0 = pure, 1 = max diversity)."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        cat_cols = [c for c in df.columns if df[c].nunique() <= 20]
    rows = []
    for c in cat_cols:
        probs = df[c].value_counts(normalize=True).values
        gini = round(float(1 - np.sum(probs ** 2)), 4)
        rows.append({"column": c, "gini": gini, "unique": int(df[c].nunique())})
    result = pd.DataFrame(rows).sort_values("gini", ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, summary="Gini impurity per categorical column")
