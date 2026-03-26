"""handle_top_bottom_values handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_top_bottom_values(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Show top N and bottom N values of a column."""
    col = params.get("column")
    n = params.get("n", 5)
    if not col or col not in df.columns:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = num_cols[0] if num_cols else df.columns[0]

    top = df.nlargest(n, col) if pd.api.types.is_numeric_dtype(df[col]) else df.head(n)
    bottom = df.nsmallest(n, col) if pd.api.types.is_numeric_dtype(df[col]) else df.tail(n)
    top = top.copy()
    bottom = bottom.copy()
    top["_group"] = "top"
    bottom["_group"] = "bottom"
    result = pd.concat([top, bottom], ignore_index=True)
    return HandlerResult(success=True, result_df=result,
                         summary=f"Top {n} and bottom {n} values by '{col}'")
