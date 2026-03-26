"""handle_scale_minmax handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_scale_minmax(df: pd.DataFrame, params: dict) -> HandlerResult:
    result = df.copy()
    num_cols = result.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns to scale")
    result[num_cols] = MinMaxScaler().fit_transform(result[num_cols])
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"MinMax scaled {len(num_cols)} numeric columns to [0,1]")
