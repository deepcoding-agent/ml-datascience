"""handle_dtypes handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_dtypes(df: pd.DataFrame, params: dict) -> HandlerResult:
    result = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(d) for d in df.dtypes.values],
        "null_pct": (df.isnull().mean() * 100).round(1).values,
        "unique": [df[c].nunique() for c in df.columns],
    })
    return HandlerResult(success=True, result_df=result, summary="Column data types")
