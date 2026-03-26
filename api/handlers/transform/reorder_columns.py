"""handle_reorder_columns handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_reorder_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Reorder columns alphabetically or by a provided list."""
    order = params.get("order")  # list of column names
    result = df.copy()

    if order and isinstance(order, list):
        valid = [c for c in order if c in result.columns]
        remaining = [c for c in result.columns if c not in valid]
        result = result[valid + remaining]
        desc = f"custom order ({len(valid)} specified)"
    else:
        result = result.reindex(sorted(result.columns), axis=1)
        desc = "alphabetical"

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Reordered {len(result.columns)} columns ({desc})",
    )
