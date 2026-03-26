"""handle_cramers_v handler."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger
log = get_logger(__name__)

def handle_cramers_v(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compute Cramer's V association between categorical columns."""
    cats = BaseHandler.get_categorical_cols(df)
    if len(cats) < 2:
        return HandlerResult(success=False, error="Need at least 2 categorical columns")
    cats = [c for c in cats if df[c].nunique() <= 50][:10]
    rows = []
    for i, c1 in enumerate(cats):
        for c2 in cats[i+1:]:
            ct = pd.crosstab(df[c1], df[c2])
            chi2 = chi2_contingency(ct)[0]
            n = ct.sum().sum()
            k = min(ct.shape) - 1
            v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0
            strength = "strong" if v >= 0.5 else "moderate" if v >= 0.3 else "weak"
            rows.append({"col_a": c1, "col_b": c2, "cramers_v": round(v, 4), "strength": strength})
    rows.sort(key=lambda r: r["cramers_v"], reverse=True)
    result = pd.DataFrame(rows)
    return HandlerResult(success=True, result_df=result, output_type="query",
                         summary=f"Cramer's V for {len(rows)} categorical pairs. Strongest: {rows[0]['col_a']}↔{rows[0]['col_b']} (V={rows[0]['cramers_v']})" if rows else "No pairs")
