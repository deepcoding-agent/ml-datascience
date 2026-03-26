"""handle_duplicate_column handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_duplicate_column(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Duplicate a column with a new name."""
    col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
    if err:
        return err
    new_name = params.get("new_name", f"{col}_copy")
    result = df.copy()
    result[new_name] = result[col]
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Duplicated '{col}' → '{new_name}'",
    )
