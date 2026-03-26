"""handle_scale_robust handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_scale_robust(df: pd.DataFrame, params: dict) -> HandlerResult:
    """RobustScaler — scales using median/IQR, resistant to outliers."""
    result = df.copy()
    num_cols = result.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns to scale")
    result[num_cols] = RobustScaler().fit_transform(result[num_cols])
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Robust scaled {len(num_cols)} numeric columns (median-centered, IQR-scaled)")
