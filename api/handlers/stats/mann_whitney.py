"""handle_mann_whitney handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_mann_whitney(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Mann-Whitney U test (non-parametric alternative to t-test)."""
    from scipy import stats as sp_stats

    col = params.get("column")
    group_col = params.get("group_column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not col or col not in df.columns:
        col = num_cols[0] if num_cols else None
    if not group_col or group_col not in df.columns:
        group_col = next((c for c in cat_cols if df[c].nunique() == 2), cat_cols[0] if cat_cols else None)
    if not col or not group_col:
        return HandlerResult(success=False, error="Need a numeric column and a categorical group column")

    groups = df[group_col].dropna().unique()[:2]
    if len(groups) < 2:
        return HandlerResult(success=False, error=f"Column '{group_col}' needs at least 2 groups")

    g1 = df.loc[df[group_col] == groups[0], col].dropna()
    g2 = df.loc[df[group_col] == groups[1], col].dropna()
    stat, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
    result = pd.DataFrame([{
        "numeric_col": col, "group_col": group_col,
        "group_1": str(groups[0]), "group_2": str(groups[1]),
        "u_stat": round(float(stat), 4), "p_value": round(p, 6),
        "significant": "Yes" if p < 0.05 else "No",
    }])
    return HandlerResult(success=True, result_df=result,
                         summary=f"Mann-Whitney U {col} by {group_col}: U={stat:.1f}, p={p:.4f}")
