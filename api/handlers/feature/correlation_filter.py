"""handle_correlation_filter handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_correlation_filter(df: pd.DataFrame, params: dict) -> HandlerResult:
    threshold = params.get("value", 0.95)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return HandlerResult(success=False, error="Need at least 2 numeric columns")
    corr = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [c for c in upper.columns if upper[c].max() > threshold]
    result = df.drop(columns=to_drop)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Dropped {len(to_drop)} columns with correlation > {threshold}: {to_drop}")
