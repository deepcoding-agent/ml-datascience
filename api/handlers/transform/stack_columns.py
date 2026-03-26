"""handle_stack_columns handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_stack_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Stack selected columns into long format with col_name and col_value."""
    columns = params.get("columns", [])
    if not columns or not isinstance(columns, list):
        columns = df.columns.tolist()

    valid = [c for c in columns if c in df.columns]
    if not valid:
        return HandlerResult(success=False, error="No valid columns to stack")

    id_cols = [c for c in df.columns if c not in valid]
    result = df.melt(id_vars=id_cols if id_cols else None,
                     value_vars=valid,
                     var_name="col_name", value_name="col_value")

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Stacked {len(valid)} columns → long format ({len(result):,} rows)",
    )
