"""handle_coefficient_variation handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_coefficient_variation(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Coefficient of variation (CV = std/mean) per numeric column."""
    num_cols = df.select_dtypes(include="number").columns
    rows = []
    for c in num_cols:
        mean = float(df[c].mean())
        std = float(df[c].std())
        cv = round(std / mean * 100, 2) if mean != 0 else float("inf")
        rows.append({"column": c, "mean": round(mean, 4), "std": round(std, 4), "cv_pct": cv})
    result = pd.DataFrame(rows).sort_values("cv_pct", ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, summary="Coefficient of variation per column")
