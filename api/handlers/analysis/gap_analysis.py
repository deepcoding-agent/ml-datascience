"""handle_gap_analysis handler."""
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


def handle_gap_analysis(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in num_cols else (num_cols[0] if num_cols else None)
    if col is None:
        return HandlerResult(success=False, error="No numeric column found")
    s = df[col].dropna().sort_values().reset_index(drop=True)
    diffs = s.diff().dropna()
    mean_gap = float(diffs.mean())
    std_gap = float(diffs.std()) if len(diffs) > 1 else 0
    threshold = mean_gap + 2 * std_gap
    large_gaps = diffs[diffs > threshold]
    rows = [{"position": int(i), "value_before": round(float(s.iloc[i - 1]), 4), "value_after": round(float(s.iloc[i]), 4),
             "gap_size": round(float(diffs.iloc[i - 1]), 4)} for i in large_gaps.index[:20]]
    result_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["position", "value_before", "value_after", "gap_size"])
    return HandlerResult(success=True, result_df=result_df, output_type="query",
                         summary=f"Found {len(large_gaps)} significant gaps in '{col}' (threshold={threshold:.2f}, mean_gap={mean_gap:.2f})")
