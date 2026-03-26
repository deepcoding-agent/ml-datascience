"""handle_top_correlations handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_top_correlations(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Top N most correlated feature pairs (easier to read than full matrix)."""
    n = params.get("n", 10)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")

    corr = df[num_cols].corr().abs()
    # Extract upper triangle pairs
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append({
                "feature_1": corr.columns[i],
                "feature_2": corr.columns[j],
                "correlation": round(float(df[num_cols].corr().iloc[i, j]), 4),
                "abs_correlation": round(float(corr.iloc[i, j]), 4),
            })
    result = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False).head(n).reset_index(drop=True)

    return HandlerResult(
        success=True, result_df=result,
        summary=f"Top {min(n, len(result))} correlated pairs (highest: {result.iloc[0]['feature_1']} ↔ {result.iloc[0]['feature_2']} = {result.iloc[0]['correlation']})" if len(result) > 0 else "No pairs",
    )
