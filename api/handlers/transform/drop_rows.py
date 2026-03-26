"""handle_drop_rows handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_drop_rows(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Drop rows by index range or specific indices."""
    start = params.get("start")
    end = params.get("end")
    indices = params.get("indices")  # list of int
    result = df.copy()
    original = len(result)

    if indices and isinstance(indices, list):
        result = result.drop(index=[i for i in indices if i in result.index]).reset_index(drop=True)
    elif start is not None and end is not None:
        result = result.drop(index=range(int(start), int(end) + 1), errors="ignore").reset_index(drop=True)
    elif start is not None:
        result = result.iloc[int(start):].reset_index(drop=True)
    else:
        return HandlerResult(success=False, error="Specify start/end range or indices list")

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Dropped rows: {original:,} → {len(result):,} ({original - len(result)} removed)",
    )
