"""handle_time_stats handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_time_stats(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Time-series specific stats (autocorrelation, stationarity hint)."""
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not col or col not in df.columns:
        col = num_cols[0] if num_cols else None
    if not col:
        return HandlerResult(success=False, error="Need a numeric column for time-series stats")

    data = df[col].dropna()
    rows = []
    for lag in [1, 2, 5, 10]:
        if lag < len(data):
            ac = round(float(data.autocorr(lag=lag)), 4)
            rows.append({"lag": lag, "autocorrelation": ac})

    # Simple stationarity hint: compare first/second half means
    mid = len(data) // 2
    m1, m2 = float(data.iloc[:mid].mean()), float(data.iloc[mid:].mean())
    drift = abs(m1 - m2) / max(abs(m1), 1e-10)
    stationary = "Likely stationary" if drift < 0.1 else "Non-stationary hint"

    result = pd.DataFrame(rows) if rows else pd.DataFrame([{"info": "Not enough data"}])
    return HandlerResult(success=True, result_df=result,
                         summary=f"Time stats for '{col}': {stationary}, first/second mean drift={drift:.2%}")
