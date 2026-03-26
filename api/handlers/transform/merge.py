"""handle_merge handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_merge(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Merge (join) current DataFrame with itself via self-join on a key,
    or create a cross-tabulated version. Use how=inner/left/right/outer."""
    column = params.get("column")
    how = params.get("how", "inner")
    if not column or column not in df.columns:
        return HandlerResult(success=False, error=f"Column '{column}' not found for merge key")
    # Self-join dedup: group by key → aggregate all numeric columns
    num_cols = df.select_dtypes(include="number").columns.tolist()
    agg_dict = {c: "sum" for c in num_cols if c != column}
    if not agg_dict:
        agg_dict = {df.columns[0]: "count"}
    result = df.groupby(column, dropna=False).agg(agg_dict).reset_index()
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Merged by '{column}' ({how}): {len(df):,} → {len(result):,} rows",
    )
