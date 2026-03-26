"""handle_scale_standard handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_scale_standard(df: pd.DataFrame, params: dict) -> HandlerResult:
    result = df.copy()
    num_cols = result.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns to scale")
    result[num_cols] = StandardScaler().fit_transform(result[num_cols])
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Standard scaled {len(num_cols)} numeric columns (mean=0, std=1)")
