"""handle_concat_columns handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_concat_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Concatenate string columns into a new column."""
    cols = params.get("columns", [])
    separator = params.get("separator", "_")
    new_name = params.get("new_name", "combined")
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cols = cat_cols[:2] if len(cat_cols) >= 2 else df.columns[:2].tolist()
    result = df.copy()
    result[new_name] = result[cols[0]].astype(str)
    for c in cols[1:]:
        result[new_name] = result[new_name] + separator + result[c].astype(str)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Concatenated {cols} → '{new_name}'")
