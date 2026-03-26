"""handle_transpose handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_transpose(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Transpose the DataFrame (swap rows and columns)."""
    result = df.T.reset_index()
    result.columns = ["feature"] + [f"row_{i}" for i in range(len(result.columns) - 1)]
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Transposed: {df.shape[0]}×{df.shape[1]} → {result.shape[0]}×{result.shape[1]}",
    )
