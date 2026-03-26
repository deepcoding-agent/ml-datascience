"""handle_sort handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_sort(df: pd.DataFrame, params: dict) -> HandlerResult:
    col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
    if err:
        return err
    ascending = params.get("ascending", True)
    result = df.sort_values(col, ascending=ascending).reset_index(drop=True)
    order = "ascending" if ascending else "descending"
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Sorted by '{col}' {order}")
