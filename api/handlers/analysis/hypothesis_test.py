"""handle_hypothesis_test handler."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_hypothesis_test(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Auto-choose t-test or Mann-Whitney based on normality."""
    from scipy import stats as sp_stats

    col = params.get("column")
    group_col = params.get("group_column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not col or col not in num_cols:
        col = num_cols[0] if num_cols else None
    if col is None:
        return HandlerResult(success=False, error="No numeric column found")

    if not group_col or group_col not in cat_cols:
        for c in cat_cols:
            if df[c].nunique() == 2:
                group_col = c
                break
        if not group_col:
            group_col = cat_cols[0] if cat_cols else None
    if group_col is None:
        return HandlerResult(success=False, error="No categorical column found for grouping")

    groups = df[group_col].dropna().unique()[:2]
    if len(groups) < 2:
        return HandlerResult(success=False, error=f"Need at least 2 groups in '{group_col}', found {len(groups)}")

    g1 = df[df[group_col] == groups[0]][col].dropna()
    g2 = df[df[group_col] == groups[1]][col].dropna()

    normal_1 = sp_stats.shapiro(g1.sample(min(len(g1), 5000), random_state=42))[1] > 0.05 if len(g1) >= 8 else False
    normal_2 = sp_stats.shapiro(g2.sample(min(len(g2), 5000), random_state=42))[1] > 0.05 if len(g2) >= 8 else False

    if normal_1 and normal_2:
        stat, p = sp_stats.ttest_ind(g1, g2)
        test_name = "Independent t-test"
    else:
        stat, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
        test_name = "Mann-Whitney U"

    effect_size = abs(g1.mean() - g2.mean()) / max(float(pd.concat([g1, g2]).std()), 1e-10)
    sig = "Significant" if p < 0.05 else "Not significant"

    result_df = pd.DataFrame({
        "metric": ["test", "statistic", "p_value", "significant", "effect_size_d",
                   f"mean_{groups[0]}", f"mean_{groups[1]}", "n_group_1", "n_group_2"],
        "value": [test_name, round(float(stat), 4), round(float(p), 6), sig,
                  round(float(effect_size), 3), round(float(g1.mean()), 3),
                  round(float(g2.mean()), 3), len(g1), len(g2)],
    })

    fig = px.box(df[df[group_col].isin(groups)], x=group_col, y=col, color=group_col)
    _style(fig, title=f"{test_name}: {col} by {group_col} (p={float(p):.4f}, {sig})")

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"{test_name}: {sig} (p={float(p):.4f}). {groups[0]} mean={g1.mean():.3f} vs {groups[1]} mean={g2.mean():.3f}, Cohen's d={effect_size:.3f}",
    )
