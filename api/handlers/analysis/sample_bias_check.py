"""handle_sample_bias_check handler."""
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


def handle_sample_bias_check(df: pd.DataFrame, params: dict) -> HandlerResult:
    sample_frac = float(params.get("sample_frac", 0.3))
    num_cols = df.select_dtypes(include="number").columns.tolist()[:8]
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns")
    sample = df.sample(frac=sample_frac, random_state=42)
    rows = []
    for c in num_cols:
        pop_mean, samp_mean = float(df[c].mean()), float(sample[c].mean())
        bias = abs(pop_mean - samp_mean) / max(abs(pop_mean), 1e-10) * 100
        rows.append({"column": c, "population_mean": round(pop_mean, 4), "sample_mean": round(samp_mean, 4),
                      "bias_pct": round(bias, 2), "acceptable": bias < 5})
    result_df = pd.DataFrame(rows)
    issues = sum(1 for r in rows if not r["acceptable"])
    return HandlerResult(success=True, result_df=result_df, output_type="query",
                         summary=f"Sample bias check ({sample_frac*100:.0f}% sample): {issues}/{len(rows)} features show >5% bias")
