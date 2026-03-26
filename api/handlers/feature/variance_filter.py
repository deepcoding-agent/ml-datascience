"""handle_variance_filter handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_variance_filter(df: pd.DataFrame, params: dict) -> HandlerResult:
    threshold = params.get("value", 0.01)
    num_cols = df.select_dtypes(include="number").columns
    variances = df[num_cols].var()
    low_var = variances[variances < threshold].index.tolist()
    result = df.drop(columns=low_var)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Dropped {len(low_var)} low-variance columns (threshold={threshold}): {low_var}")
