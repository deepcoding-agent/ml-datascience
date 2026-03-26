"""handle_ks_test handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_ks_test(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Kolmogorov-Smirnov normality test per numeric column."""
    from scipy import stats as sp_stats

    col = params.get("column")
    num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()
    rows = []
    for c in num_cols:
        data = df[c].dropna()
        if len(data) < 3:
            continue
        normed = (data - data.mean()) / data.std() if data.std() > 0 else data
        stat, p = sp_stats.kstest(normed, "norm")
        rows.append({
            "column": c, "ks_stat": round(stat, 4), "p_value": round(p, 6),
            "normal": "Yes" if p > 0.05 else "No",
        })
    result = pd.DataFrame(rows)
    normal_n = sum(1 for r in rows if r["normal"] == "Yes")
    return HandlerResult(success=True, result_df=result,
                         summary=f"KS normality test: {normal_n}/{len(rows)} columns appear normal")
