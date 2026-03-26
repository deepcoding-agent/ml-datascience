"""handle_distribution_fit handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_distribution_fit(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Fit best distribution (normal/lognormal/exponential) to a numeric column."""
    from scipy import stats as sp_stats

    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not col or col not in df.columns:
        col = num_cols[0] if num_cols else None
    if not col:
        return HandlerResult(success=False, error="Need a numeric column")

    data = df[col].dropna().values
    results = []
    for name, dist in [("normal", sp_stats.norm), ("lognormal", sp_stats.lognorm),
                       ("exponential", sp_stats.expon)]:
        try:
            params_fit = dist.fit(data)
            ks_stat, p = sp_stats.kstest(data, dist.cdf, args=params_fit)
            results.append({"distribution": name, "ks_stat": round(ks_stat, 4),
                            "p_value": round(p, 6)})
        except Exception:
            results.append({"distribution": name, "ks_stat": None, "p_value": None})

    result = pd.DataFrame(results).sort_values("p_value", ascending=False).reset_index(drop=True)
    best = result.iloc[0]["distribution"] if len(result) > 0 else "unknown"
    return HandlerResult(success=True, result_df=result,
                         summary=f"Best fit for '{col}': {best}")
