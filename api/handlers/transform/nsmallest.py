"""handle_nsmallest handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_nsmallest(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Bottom N rows by column value."""
    col = params.get("column")
    n = params.get("n", 10)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
    result = df.nsmallest(min(n, len(df)), col)
    return HandlerResult(success=True, result_df=result, output_type="query",
                         summary=f"Bottom {min(n, len(df))} rows by '{col}'")
