"""handle_seasonality_detect handler."""
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


def handle_seasonality_detect(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in num_cols else (num_cols[0] if num_cols else None)
    if col is None:
        return HandlerResult(success=False, error="No numeric column found")
    s = df[col].dropna().reset_index(drop=True)
    if len(s) < 10:
        return HandlerResult(success=False, error="Need ≥10 data points")
    autocorrs = [round(float(s.autocorr(lag=i)), 4) for i in range(1, min(len(s) // 2, 30))]
    peak_lag = int(np.argmax(autocorrs[1:]) + 2) if len(autocorrs) > 1 else 0
    peak_val = max(autocorrs[1:]) if len(autocorrs) > 1 else 0
    result_df = pd.DataFrame({"lag": list(range(1, len(autocorrs) + 1)), "autocorrelation": autocorrs})
    fig = px.bar(result_df, x="lag", y="autocorrelation", text="autocorrelation")
    fig.update_traces(marker_color="#FB8C3C", texttemplate="%{text:.3f}", textposition="outside")
    _style(fig, title=f"Autocorrelation — {col} (peak lag={peak_lag}, r={peak_val:.3f})")
    seasonal = "Likely seasonal" if peak_val > 0.3 else "No clear seasonality"
    return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"{seasonal} in '{col}'. Peak autocorrelation at lag {peak_lag} (r={peak_val:.3f})")
