"""handle_resample handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_resample(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Resample time series data to a different frequency (daily, weekly, monthly)."""
    date_col = params.get("column")
    freq = params.get("freq", "M")  # D, W, M, Q, Y
    agg = params.get("agg", "mean")

    # Find datetime column
    dt_cols = df.select_dtypes(include="datetime").columns.tolist()
    if date_col and date_col in df.columns:
        target = date_col
    elif dt_cols:
        target = dt_cols[0]
    else:
        # Try to parse object columns as dates
        for c in df.select_dtypes(include="object").columns:
            try:
                df[c] = pd.to_datetime(df[c], format="mixed")
                target = c
                break
            except Exception:
                continue
        else:
            return HandlerResult(success=False, error="No datetime column found for resampling")

    result = df.copy()
    result[target] = pd.to_datetime(result[target], format="mixed", errors="coerce")
    result = result.set_index(target)

    num_cols = result.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return HandlerResult(success=False, error="No numeric columns to aggregate")

    freq_map = {"daily": "D", "weekly": "W", "monthly": "ME", "quarterly": "QE", "yearly": "YE",
                "M": "ME", "Q": "QE", "Y": "YE"}
    freq = freq_map.get(freq.lower(), freq)

    try:
        resampled = result[num_cols].resample(freq).agg(agg).reset_index()
        return HandlerResult(
            success=True, result_df=resampled, output_type="generate",
            summary=f"Resampled by '{target}' to {freq} ({agg}): {len(df):,} → {len(resampled):,} rows",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Resample error: {e}")
