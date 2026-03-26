"""handle_chi2_test handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_chi2_test(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Chi-squared independence test between two categorical columns."""
    from scipy import stats as sp_stats

    cols = params.get("columns", [])
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col1 = cols[0] if len(cols) > 0 and cols[0] in df.columns else (cat_cols[0] if len(cat_cols) > 0 else None)
    col2 = cols[1] if len(cols) > 1 and cols[1] in df.columns else (cat_cols[1] if len(cat_cols) > 1 else None)
    if not col1 or not col2:
        return HandlerResult(success=False, error="Need 2 categorical columns for chi-squared test")

    ct = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = sp_stats.chi2_contingency(ct)
    result = pd.DataFrame([{
        "column_1": col1, "column_2": col2,
        "chi2": round(chi2, 4), "p_value": round(p, 6),
        "dof": dof, "significant": "Yes" if p < 0.05 else "No",
    }])
    verdict = "significant association" if p < 0.05 else "no significant association"
    return HandlerResult(success=True, result_df=result,
                         summary=f"Chi2 test {col1} vs {col2}: chi2={chi2:.2f}, p={p:.4f} — {verdict}")
