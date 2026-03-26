"""handle_split_column handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_split_column(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Split a string column by delimiter into multiple columns."""
    col = params.get("column")
    delimiter = params.get("delimiter", ",")
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    result = df.copy()
    split_df = result[col].astype(str).str.split(delimiter, expand=True)
    split_df.columns = [f"{col}_{i+1}" for i in range(split_df.shape[1])]
    for c in split_df.columns:
        split_df[c] = split_df[c].str.strip()
    result = pd.concat([result, split_df], axis=1)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Split '{col}' by '{delimiter}' → {split_df.shape[1]} new columns")
