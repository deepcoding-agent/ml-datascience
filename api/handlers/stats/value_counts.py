"""handle_value_counts handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_value_counts(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    if not col or col not in df.columns:
        # Pick first categorical or first column
        cats = df.select_dtypes(include=["object", "category"]).columns
        col = cats[0] if len(cats) > 0 else df.columns[0]
    n = params.get("n", 10)
    vc = df[col].value_counts().head(n).reset_index()
    vc.columns = [col, "count"]
    total = int(df[col].count())
    vc["percentage"] = (vc["count"] / max(total, 1) * 100).round(2)
    return HandlerResult(success=True, result_df=vc, summary=f"Top {n} values in '{col}' (total={total:,})")
