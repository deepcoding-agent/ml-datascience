"""handle_ab_test handler."""
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


def handle_ab_test(df: pd.DataFrame, params: dict) -> HandlerResult:
    """A/B test with significance: p-value, effect size, confidence interval."""
    from scipy import stats as sp_stats

    metric_col = params.get("column") or params.get("metric")
    group_col = params.get("group_column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not metric_col or metric_col not in num_cols:
        metric_col = num_cols[0] if num_cols else None
    if metric_col is None:
        return HandlerResult(success=False, error="No numeric metric column found")

    if not group_col or group_col not in cat_cols:
        for c in cat_cols:
            if df[c].nunique() == 2:
                group_col = c
                break
        if not group_col:
            group_col = cat_cols[0] if cat_cols else None
    if group_col is None:
        return HandlerResult(success=False, error="No categorical group column found")

    groups = df[group_col].dropna().unique()[:2]
    if len(groups) < 2:
        return HandlerResult(success=False, error="Need exactly 2 groups for A/B test")

    a = df[df[group_col] == groups[0]][metric_col].dropna()
    b = df[df[group_col] == groups[1]][metric_col].dropna()

    stat, p = sp_stats.ttest_ind(a, b)
    pooled_std = float(np.sqrt((a.var() * (len(a) - 1) + b.var() * (len(b) - 1)) / (len(a) + len(b) - 2)))
    effect_d = abs(float(a.mean()) - float(b.mean())) / max(pooled_std, 1e-10)
    se_diff = float(np.sqrt(a.var() / len(a) + b.var() / len(b)))
    ci_low = float(a.mean() - b.mean()) - 1.96 * se_diff
    ci_high = float(a.mean() - b.mean()) + 1.96 * se_diff

    winner = str(groups[0]) if float(a.mean()) > float(b.mean()) else str(groups[1])
    sig = "Significant" if p < 0.05 else "Not significant"

    result_df = pd.DataFrame({
        "metric": ["test", "p_value", "significant", "effect_size_d", f"mean_{groups[0]}", f"mean_{groups[1]}",
                   "diff", "ci_lower", "ci_upper", "winner", "n_A", "n_B"],
        "value": ["Welch t-test", round(float(p), 6), sig, round(effect_d, 3),
                  round(float(a.mean()), 3), round(float(b.mean()), 3),
                  round(float(a.mean() - b.mean()), 3), round(ci_low, 3), round(ci_high, 3),
                  winner, len(a), len(b)],
    })

    fig = px.box(df[df[group_col].isin(groups)], x=group_col, y=metric_col, color=group_col)
    _style(fig, title=f"A/B Test: {metric_col} — {sig} (p={float(p):.4f})")
    fig.update_layout(showlegend=False)

    return HandlerResult(
        success=True, result_df=result_df, output_type="query",
        charts_plotly=[fig.to_json()],
        summary=f"A/B test ({metric_col}): {sig} (p={float(p):.4f}). {winner} wins. Effect size d={effect_d:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]",
    )
