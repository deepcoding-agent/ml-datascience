"""handle_encode_ordinal handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_encode_ordinal(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Ordinal-encode a categorical column using a custom order list.
    If no order is given, sorts unique values alphabetically."""
    col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
    if err:
        return err
    order = params.get("order")  # list of strings
    result = df.copy()

    if order and isinstance(order, list):
        mapping = {v: i for i, v in enumerate(order)}
    else:
        uniques = sorted(result[col].dropna().unique(), key=str)
        mapping = {v: i for i, v in enumerate(uniques)}

    result[f"{col}_ordinal"] = result[col].map(mapping)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Ordinal-encoded '{col}' → '{col}_ordinal' ({len(mapping)} levels)",
    )
