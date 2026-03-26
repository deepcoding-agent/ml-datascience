"""handle_inject_null handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_inject_null(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Inject random NaN values into a copy of the DataFrame."""
    fraction = params.get("value", 15)
    if fraction > 1:
        fraction = fraction / 100.0  # convert 15 → 0.15
    result = df.copy()
    for col in result.columns:
        n_nulls = int(len(result) * fraction)
        if n_nulls > 0:
            null_indices = np.random.choice(result.index, size=n_nulls, replace=False)
            result.loc[null_indices, col] = np.nan
    total_nulls = int(result.isnull().sum().sum())
    total_cells = result.shape[0] * result.shape[1]
    actual_pct = total_nulls / total_cells * 100 if total_cells > 0 else 0
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Injected ~{fraction*100:.0f}% null values. Actual: {actual_pct:.1f}% ({total_nulls:,}/{total_cells:,} cells)",
    )
