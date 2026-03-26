"""handle_qcut handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_qcut(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Quantile-based binning (equal-frequency bins)."""
    col = params.get("column")
    q = params.get("n", 4)
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    result = df.copy()
    try:
        result[f"{col}_qbin"] = pd.qcut(result[col], q=q, labels=False, duplicates="drop")
        actual_bins = result[f"{col}_qbin"].nunique()
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Quantile-binned '{col}' into {actual_bins} bins → '{col}_qbin'")
    except Exception as e:
        return HandlerResult(success=False, error=f"Quantile binning error: {e}")
