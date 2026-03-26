"""handle_shift_column handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_shift_column(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Shift column values up or down by N rows (like lag/lead)."""
    col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
    if err:
        return err
    periods = int(params.get("periods", 1))
    result = df.copy()
    result[f"{col}_shift_{periods}"] = result[col].shift(periods)
    direction = "down" if periods > 0 else "up"
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Shifted '{col}' {direction} by {abs(periods)} rows → '{col}_shift_{periods}'",
    )
