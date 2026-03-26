"""handle_unique_values handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_unique_values(df: pd.DataFrame, params: dict) -> HandlerResult:
    result = pd.DataFrame({
        "column": df.columns,
        "unique_count": [df[c].nunique() for c in df.columns],
        "dtype": [str(d) for d in df.dtypes.values],
    }).sort_values("unique_count", ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, summary="Unique value counts per column")
