"""handle_data_sample handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_data_sample(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Random sample with basic stats summary."""
    n = params.get("n", 10)
    n = min(n, len(df))
    sample = df.sample(n, random_state=42)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    null_total = int(df.isna().sum().sum())
    return HandlerResult(success=True, result_df=sample,
                         summary=f"Random sample of {n} rows | "
                                 f"{df.shape[0]} total rows, {len(num_cols)} numeric cols, "
                                 f"{null_total} nulls")
