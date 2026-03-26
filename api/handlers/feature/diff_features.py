"""handle_diff_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_diff_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create difference features (first/second order) for time series."""
    col = params.get("column")
    periods = params.get("periods", 1)
    result = df.copy()

    cols_to_diff = (
        [col] if col and col in result.columns
        else result.select_dtypes(include="number").columns.tolist()[:3]
    )
    created = []

    for c in cols_to_diff:
        result[f"{c}_diff{periods}"] = result[c].diff(periods)
        created.append(f"{c}_diff{periods}")

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} diff features (periods={periods})",
    )
