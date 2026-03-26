"""handle_rank handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_rank(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Add rank column based on a numeric column."""
    col = params.get("column")
    ascending = params.get("ascending", True)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
    if not col:
        return HandlerResult(success=False, error="No column to rank")
    result = df.copy()
    result[f"{col}_rank"] = result[col].rank(ascending=ascending, method="min").astype(int)
    result = result.sort_values(f"{col}_rank").reset_index(drop=True)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Ranked by '{col}' {'ascending' if ascending else 'descending'}")
