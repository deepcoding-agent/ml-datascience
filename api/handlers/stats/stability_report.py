"""handle_stability_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_stability_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Check feature stability: split data in half, compare stats."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    mid = len(df) // 2
    first_half, second_half = df.iloc[:mid], df.iloc[mid:]
    rows = []
    for c in num_cols:
        m1, m2 = float(first_half[c].mean()), float(second_half[c].mean())
        s1, s2 = float(first_half[c].std()), float(second_half[c].std())
        mean_drift = abs(m1 - m2) / max(abs(m1), 1e-10)
        rows.append({
            "column": c, "mean_first": round(m1, 4), "mean_second": round(m2, 4),
            "std_first": round(s1, 4), "std_second": round(s2, 4),
            "mean_drift_pct": round(mean_drift * 100, 2),
            "stable": "Yes" if mean_drift < 0.1 else "No",
        })
    result = pd.DataFrame(rows).sort_values("mean_drift_pct", ascending=False).reset_index(drop=True)
    stable_n = sum(1 for r in rows if r["stable"] == "Yes")
    return HandlerResult(success=True, result_df=result,
                         summary=f"Stability: {stable_n}/{len(rows)} columns stable (drift < 10%)")
