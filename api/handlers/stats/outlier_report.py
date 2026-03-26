"""handle_outlier_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_outlier_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    num_cols = df.select_dtypes(include="number").columns
    rows = []
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        rows.append({"column": col, "outlier_count": count, "lower_bound": round(lower, 2), "upper_bound": round(upper, 2)})
    result = pd.DataFrame(rows).sort_values("outlier_count", ascending=False).reset_index(drop=True)
    total = result["outlier_count"].sum()
    return HandlerResult(success=True, result_df=result, summary=f"Total outliers (IQR): {total:,}")
