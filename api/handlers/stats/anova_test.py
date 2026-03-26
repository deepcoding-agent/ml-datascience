"""handle_anova_test handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_anova_test(df: pd.DataFrame, params: dict) -> HandlerResult:
    """One-way ANOVA: compare numeric column across multiple groups."""
    from scipy import stats as sp_stats

    col = params.get("column")
    group_col = params.get("group_column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not col or col not in df.columns:
        col = num_cols[0] if num_cols else None
    if not group_col or group_col not in df.columns:
        group_col = cat_cols[0] if cat_cols else None
    if not col or not group_col:
        return HandlerResult(success=False, error="Need a numeric column and a categorical group column")

    groups = [g.dropna().values for _, g in df.groupby(group_col)[col] if len(g.dropna()) > 0]
    if len(groups) < 2:
        return HandlerResult(success=False, error="Need at least 2 groups for ANOVA")

    f_stat, p = sp_stats.f_oneway(*groups)
    result = pd.DataFrame([{
        "numeric_col": col, "group_col": group_col, "n_groups": len(groups),
        "f_stat": round(f_stat, 4), "p_value": round(p, 6),
        "significant": "Yes" if p < 0.05 else "No",
    }])
    return HandlerResult(success=True, result_df=result,
                         summary=f"ANOVA {col} by {group_col}: F={f_stat:.3f}, p={p:.4f} ({len(groups)} groups)")
