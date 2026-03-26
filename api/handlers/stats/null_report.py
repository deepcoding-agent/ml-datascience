"""handle_null_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_null_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().mean() * 100).round(2)
    result = pd.DataFrame({
        "column": df.columns,
        "null_count": null_counts.values,
        "null_pct": null_pct.values,
        "dtype": [str(d) for d in df.dtypes.values],
    }).sort_values("null_count", ascending=False).reset_index(drop=True)
    total = int(null_counts.sum())
    return HandlerResult(success=True, result_df=result, summary=f"Total nulls: {total:,} across {(null_counts > 0).sum()} columns")
