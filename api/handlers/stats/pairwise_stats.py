"""handle_pairwise_stats handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_pairwise_stats(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Pairwise statistical comparison between numeric columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")
    rows = []
    for i in range(min(len(num_cols), 10)):
        for j in range(i + 1, min(len(num_cols), 10)):
            a, b = num_cols[i], num_cols[j]
            corr = float(df[[a, b]].corr().iloc[0, 1])
            diff = float(df[a].mean() - df[b].mean())
            rows.append({
                "col_a": a, "col_b": b,
                "pearson_corr": round(corr, 4),
                "mean_diff": round(diff, 4),
                "std_ratio": round(float(df[a].std()) / max(float(df[b].std()), 1e-10), 4),
            })
    result = pd.DataFrame(rows).sort_values("pearson_corr", key=abs, ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result,
                         summary=f"Pairwise stats for {min(len(num_cols), 10)} numeric columns")
