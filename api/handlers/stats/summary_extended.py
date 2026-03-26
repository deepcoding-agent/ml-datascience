"""handle_summary_extended handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_summary_extended(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Extended summary: mean, median, mode, std, var, range, IQR, skew, kurtosis."""
    num_cols = df.select_dtypes(include="number").columns
    rows = []
    for c in num_cols:
        data = df[c].dropna()
        modes = data.mode()
        q1, q3 = float(data.quantile(0.25)), float(data.quantile(0.75))
        rows.append({
            "column": c,
            "mean": round(float(data.mean()), 4),
            "median": round(float(data.median()), 4),
            "mode": round(float(modes.iloc[0]), 4) if len(modes) > 0 else None,
            "std": round(float(data.std()), 4),
            "variance": round(float(data.var()), 4),
            "range": round(float(data.max() - data.min()), 4),
            "IQR": round(q3 - q1, 4),
            "skewness": round(float(data.skew()), 4),
            "kurtosis": round(float(data.kurtosis()), 4),
        })
    result = pd.DataFrame(rows)
    return HandlerResult(success=True, result_df=result,
                         summary=f"Extended summary for {len(num_cols)} numeric columns")
