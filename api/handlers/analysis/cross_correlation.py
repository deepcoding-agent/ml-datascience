"""handle_cross_correlation handler."""
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


def handle_cross_correlation(df: pd.DataFrame, params: dict) -> HandlerResult:
    columns = params.get("columns", [])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(columns) >= 2 and all(c in num_cols for c in columns[:2]):
        c1, c2 = columns[0], columns[1]
    elif len(num_cols) >= 2:
        c1, c2 = num_cols[0], num_cols[1]
    else:
        return HandlerResult(success=False, error="Need 2 numeric columns")
    max_lag = min(len(df) // 3, 20)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = float(df[c1].iloc[lag:].reset_index(drop=True).corr(df[c2].iloc[:len(df) - lag].reset_index(drop=True)))
        else:
            r = float(df[c2].iloc[-lag:].reset_index(drop=True).corr(df[c1].iloc[:len(df) + lag].reset_index(drop=True)))
        rows.append({"lag": lag, "correlation": round(r, 4) if not np.isnan(r) else 0})
    result_df = pd.DataFrame(rows)
    peak = result_df.loc[result_df["correlation"].abs().idxmax()]
    fig = px.bar(result_df, x="lag", y="correlation")
    fig.update_traces(marker_color="#2EC4B6")
    _style(fig, title=f"Cross-Correlation: {c1} vs {c2} (peak lag={int(peak['lag'])}, r={peak['correlation']:.3f})")
    return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"Cross-correlation {c1}↔{c2}: peak at lag {int(peak['lag'])} (r={peak['correlation']:.3f})")
