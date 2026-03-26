"""handle_normality_test handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_normality_test(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Shapiro-Wilk normality test for numeric columns."""
    from scipy import stats as sp_stats

    col = params.get("column")
    num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()
    rows = []

    for c in num_cols:
        data = df[c].dropna()
        sample = data.sample(min(5000, len(data)), random_state=42) if len(data) > 5000 else data
        try:
            stat, p_value = sp_stats.shapiro(sample)
            rows.append({
                "column": c,
                "shapiro_stat": round(stat, 4),
                "p_value": round(p_value, 6),
                "normal": "Yes" if p_value > 0.05 else "No",
                "skewness": round(float(data.skew()), 4),
                "kurtosis": round(float(data.kurtosis()), 4),
            })
        except Exception:
            rows.append({"column": c, "shapiro_stat": None, "p_value": None, "normal": "Error",
                         "skewness": round(float(data.skew()), 4), "kurtosis": round(float(data.kurtosis()), 4)})

    result = pd.DataFrame(rows)
    normal_count = sum(1 for r in rows if r["normal"] == "Yes")
    return HandlerResult(
        success=True, result_df=result,
        summary=f"Normality test: {normal_count}/{len(rows)} columns appear normal (p > 0.05)",
    )
