"""handle_zscore_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_zscore_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Add z-score columns for numeric features."""
    col = params.get("column")
    result = df.copy()
    cols = (
        [col] if col and col in result.columns
        else result.select_dtypes(include="number").columns.tolist()
    )
    created = []
    for c in cols:
        mean = result[c].mean()
        std = result[c].std()
        if std > 0:
            result[f"{c}_zscore"] = ((result[c] - mean) / std).round(4)
            created.append(f"{c}_zscore")
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} z-score feature columns",
    )
