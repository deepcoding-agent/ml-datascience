"""handle_is_zero_features handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_is_zero_features(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create binary is_zero indicator columns for numeric columns."""
    result = df.copy()
    created = []
    for c in result.select_dtypes(include="number").columns:
        if (result[c] == 0).any():
            result[f"{c}_is_zero"] = (result[c] == 0).astype(int)
            created.append(f"{c}_is_zero")
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created {len(created)} is_zero indicator columns",
    )
