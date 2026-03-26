"""handle_normality_comprehensive handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_normality_comprehensive(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Comprehensive normality: Shapiro-Wilk + D'Agostino + Anderson-Darling."""
    from scipy import stats as sp_stats

    col = params.get("column")
    num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()
    rows = []
    for c in num_cols:
        data = df[c].dropna()
        sample = data.sample(min(5000, len(data)), random_state=42) if len(data) > 5000 else data
        row: dict = {"column": c}
        # Shapiro
        try:
            sw_stat, sw_p = sp_stats.shapiro(sample)
            row["shapiro_stat"] = round(sw_stat, 4)
            row["shapiro_p"] = round(sw_p, 6)
        except Exception:
            row["shapiro_stat"] = None
            row["shapiro_p"] = None
        # D'Agostino
        try:
            da_stat, da_p = sp_stats.normaltest(sample)
            row["dagostino_stat"] = round(float(da_stat), 4)
            row["dagostino_p"] = round(float(da_p), 6)
        except Exception:
            row["dagostino_stat"] = None
            row["dagostino_p"] = None
        # Anderson-Darling
        try:
            ad = sp_stats.anderson(sample, dist="norm")
            row["anderson_stat"] = round(float(ad.statistic), 4)
            row["anderson_cv_5pct"] = round(float(ad.critical_values[2]), 4)
            row["anderson_normal"] = "Yes" if ad.statistic < ad.critical_values[2] else "No"
        except Exception:
            row["anderson_stat"] = None
            row["anderson_cv_5pct"] = None
            row["anderson_normal"] = "Error"

        # Consensus
        votes = sum([
            1 if row.get("shapiro_p") and row["shapiro_p"] > 0.05 else 0,
            1 if row.get("dagostino_p") and row["dagostino_p"] > 0.05 else 0,
            1 if row.get("anderson_normal") == "Yes" else 0,
        ])
        row["consensus"] = "Normal" if votes >= 2 else "Not normal"
        rows.append(row)

    result = pd.DataFrame(rows)
    normal_n = sum(1 for r in rows if r["consensus"] == "Normal")
    return HandlerResult(success=True, result_df=result,
                         summary=f"Comprehensive normality: {normal_n}/{len(rows)} columns normal (2/3 tests agree)")
