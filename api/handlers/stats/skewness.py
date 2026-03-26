"""handle_skewness handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_skewness(df: pd.DataFrame, params: dict) -> HandlerResult:
    num_cols = df.select_dtypes(include="number").columns
    skew = df[num_cols].skew().round(4)
    result = skew.reset_index()
    result.columns = ["column", "skewness"]
    result = result.sort_values("skewness", key=abs, ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, summary="Skewness per numeric column")
