"""handle_benchmark_compare handler."""
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


def handle_benchmark_compare(df: pd.DataFrame, params: dict) -> HandlerResult:
    benchmarks = params.get("benchmarks")  # dict {col: target_value}
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not benchmarks or not isinstance(benchmarks, dict):
        benchmarks = {c: float(df[c].mean()) for c in num_cols[:5]}
    rows = []
    for col, target in benchmarks.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            actual = float(df[col].mean())
            diff = actual - float(target)
            pct = diff / abs(float(target)) * 100 if float(target) != 0 else 0
            rows.append({"column": col, "actual": round(actual, 2), "benchmark": float(target),
                         "difference": round(diff, 2), "pct_diff": round(pct, 1),
                         "status": "Above" if diff > 0 else "Below" if diff < 0 else "Equal"})
    result_df = pd.DataFrame(rows)
    fig = px.bar(result_df, x="column", y="pct_diff", color="status", text="pct_diff",
                 color_discrete_map={"Above": "#2EC4B6", "Below": "#E71D36", "Equal": "#86868B"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    _style(fig, title="Benchmark Comparison — % Difference from Target")
    return HandlerResult(success=True, result_df=result_df, output_type="query", charts_plotly=[fig.to_json()],
                         summary=f"Compared {len(rows)} metrics against benchmarks")
