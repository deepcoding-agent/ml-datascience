"""handle_rolling_stats_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_rolling_stats_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Add rolling mean/std/min/max as new feature columns."""
    col = params.get("column")
    window = params.get("window", 3)
    result = df.copy()
    cols = (
        [col] if col and col in result.columns
        else result.select_dtypes(include="number").columns.tolist()[:3]
    )
    created = []
    for c in cols:
        for agg in ["mean", "std", "min", "max"]:
            name = f"{c}_roll{window}_{agg}"
            result[name] = result[c].rolling(window, min_periods=1).agg(agg).round(4)
            created.append(name)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} rolling stats features (window={window})",
    )
