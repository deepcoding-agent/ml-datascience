"""handle_prediction_baseline handler."""
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


def handle_prediction_baseline(df: pd.DataFrame, params: dict) -> HandlerResult:
    target = params.get("column")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target and target in df.columns:
        pass
    elif num_cols:
        target = num_cols[-1]
    elif cat_cols:
        target = cat_cols[0]
    else:
        return HandlerResult(success=False, error="No columns found")
    s = df[target].dropna()
    rows = []
    if pd.api.types.is_numeric_dtype(s):
        rows.append({"baseline": "mean", "value": round(float(s.mean()), 4), "metric": "MAE", "score": round(float((s - s.mean()).abs().mean()), 4)})
        rows.append({"baseline": "median", "value": round(float(s.median()), 4), "metric": "MAE", "score": round(float((s - s.median()).abs().mean()), 4)})
        rows.append({"baseline": "zero", "value": 0, "metric": "MAE", "score": round(float(s.abs().mean()), 4)})
    else:
        mode = s.mode().iloc[0] if len(s.mode()) > 0 else "N/A"
        acc = float((s == mode).mean())
        rows.append({"baseline": "most_frequent", "value": str(mode), "metric": "accuracy", "score": round(acc, 4)})
        rows.append({"baseline": "random", "value": "uniform", "metric": "accuracy", "score": round(1.0 / max(s.nunique(), 1), 4)})
    result_df = pd.DataFrame(rows)
    return HandlerResult(success=True, result_df=result_df, output_type="query",
                         summary=f"Baselines for '{target}': best naive = {rows[0]['baseline']} ({rows[0]['metric']}={rows[0]['score']:.4f})")
