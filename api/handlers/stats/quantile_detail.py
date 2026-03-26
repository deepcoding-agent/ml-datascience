"""handle_quantile_detail handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_quantile_detail(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Detailed quantile report: 1,5,10,25,50,75,90,95,99 percentiles."""
    col = params.get("column")
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    num_cols = [col] if col and col in df.columns else df.select_dtypes(include="number").columns.tolist()
    result = df[num_cols].quantile(qs).round(4).T.reset_index()
    result.columns = ["column"] + [f"p{int(q*100)}" for q in qs]
    return HandlerResult(success=True, result_df=result,
                         summary=f"Detailed quantile report for {len(num_cols)} columns")
