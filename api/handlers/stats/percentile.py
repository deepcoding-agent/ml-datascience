"""handle_percentile handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_percentile(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Custom percentile report for numeric columns."""
    col = params.get("column")
    quantiles = params.get("quantiles", [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()

    result = df[num_cols].quantile(quantiles).round(4).T.reset_index()
    result.columns = ["column"] + [f"p{int(q*100)}" for q in quantiles]

    return HandlerResult(
        success=True, result_df=result,
        summary=f"Percentile report for {len(num_cols)} numeric columns",
    )
