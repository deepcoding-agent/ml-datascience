"""handle_bin_column handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_bin_column(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    bins = params.get("n", 5)
    if not col or col not in df.columns:
        return HandlerResult(success=False, error=f"Column '{col}' not found")
    result = df.copy()
    result[f"{col}_bin"] = pd.cut(result[col], bins=bins, labels=False)
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Binned '{col}' into {bins} bins → '{col}_bin'")
