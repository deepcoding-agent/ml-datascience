"""handle_distance_from_mean handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_distance_from_mean(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compute distance from column mean or median."""
    col = params.get("column")
    method = params.get("method", "mean")
    result = df.copy()
    cols = (
        [col] if col and col in result.columns
        else result.select_dtypes(include="number").columns.tolist()
    )
    created = []
    for c in cols:
        center = result[c].mean() if method == "mean" else result[c].median()
        result[f"{c}_dist_{method}"] = (result[c] - center).round(4)
        created.append(f"{c}_dist_{method}")
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} distance-from-{method} features",
    )
