"""handle_pivot handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_pivot(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Pivot table: rows=index, columns=columns, values=values, aggfunc=agg."""
    index = params.get("index") or params.get("column")
    columns = params.get("columns")
    values = params.get("values")
    agg = params.get("agg", "mean")

    if not index or index not in df.columns:
        return HandlerResult(success=False, error=f"Index column '{index}' not found")

    try:
        if columns and columns in df.columns and values and values in df.columns:
            result = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=agg)
            result = result.reset_index()
            result.columns = [str(c) for c in result.columns]
        elif values and values in df.columns:
            result = pd.pivot_table(df, index=index, values=values, aggfunc=agg).reset_index()
        else:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            result = pd.pivot_table(df, index=index, values=num_cols, aggfunc=agg).reset_index()

        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"Pivot table: index='{index}', agg={agg} ({result.shape[0]}×{result.shape[1]})")
    except Exception as e:
        return HandlerResult(success=False, error=f"Pivot error: {e}")
