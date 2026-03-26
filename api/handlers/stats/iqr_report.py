"""handle_iqr_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_iqr_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Interquartile range per numeric column."""
    num_cols = df.select_dtypes(include="number").columns
    rows = []
    for c in num_cols:
        q1, q3 = float(df[c].quantile(0.25)), float(df[c].quantile(0.75))
        rows.append({"column": c, "Q1": round(q1, 4), "Q3": round(q3, 4), "IQR": round(q3 - q1, 4)})
    result = pd.DataFrame(rows).sort_values("IQR", ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, summary="IQR per numeric column")
