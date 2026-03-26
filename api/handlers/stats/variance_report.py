"""handle_variance_report handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_variance_report(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Variance per numeric column."""
    num_cols = df.select_dtypes(include="number").columns
    var = df[num_cols].var().round(4)
    result = var.reset_index()
    result.columns = ["column", "variance"]
    result = result.sort_values("variance", ascending=False).reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, summary="Variance per numeric column")
