"""handle_lag_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_lag_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create lag/shift features for time series data."""
    col = params.get("column")
    lags = params.get("lags", [1, 2, 3])
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))

    result = df.copy()
    cols_to_lag = (
        [col] if col and col in result.columns
        else result.select_dtypes(include="number").columns.tolist()[:3]
    )
    created = []

    for c in cols_to_lag:
        for lag in lags:
            result[f"{c}_lag{lag}"] = result[c].shift(lag)
            created.append(f"{c}_lag{lag}")

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} lag features from {len(cols_to_lag)} column(s), lags={lags}",
    )
