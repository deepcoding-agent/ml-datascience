"""handle_explode handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_explode(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Explode a column containing comma-separated values or lists into separate rows."""
    col = params.get("column")
    delimiter = params.get("delimiter", ",")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")

    result = df.copy()
    original = len(result)
    # Split string column into lists if needed
    if result[col].dtype == "object":
        result[col] = result[col].fillna("").astype(str).str.split(delimiter)
    result = result.explode(col, ignore_index=True)
    result[col] = result[col].astype(str).str.strip()

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Exploded '{col}': {original:,} → {len(result):,} rows",
    )
