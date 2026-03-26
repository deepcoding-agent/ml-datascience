"""handle_pct_change handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_pct_change(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Compute percentage change between consecutive rows."""
    col = params.get("column")
    periods = int(params.get("periods", 1))
    result = df.copy()

    if col and col in result.columns:
        cols = [col]
    else:
        cols = result.select_dtypes(include="number").columns.tolist()[:5]

    created: list[str] = []
    for c in cols:
        name = f"{c}_pct_change"
        result[name] = result[c].pct_change(periods=periods).round(4)
        created.append(name)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Percentage change (periods={periods}) for {len(created)} column(s)",
    )
