"""handle_ewm_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_ewm_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Exponentially weighted moving average/std as features."""
    col = params.get("column")
    span = params.get("span", 5)
    result = df.copy()
    cols = (
        [col] if col and col in result.columns
        else result.select_dtypes(include="number").columns.tolist()[:3]
    )
    created = []
    for c in cols:
        result[f"{c}_ewm_mean"] = result[c].ewm(span=span, min_periods=1).mean().round(4)
        result[f"{c}_ewm_std"] = result[c].ewm(span=span, min_periods=1).std().round(4)
        created.extend([f"{c}_ewm_mean", f"{c}_ewm_std"])
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} EWM features (span={span})",
    )
