"""handle_cross_join handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_cross_join(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Create cartesian product of unique values from two columns.
    Useful for generating all possible combinations."""
    columns = params.get("columns", [])
    if len(columns) < 2:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        columns = cat_cols[:2] if len(cat_cols) >= 2 else df.columns[:2].tolist()

    c1, c2 = columns[0], columns[1]
    if c1 not in df.columns or c2 not in df.columns:
        return HandlerResult(success=False, error=f"Columns {c1}, {c2} not found")

    vals1 = df[c1].dropna().unique()
    vals2 = df[c2].dropna().unique()
    import itertools
    combos = list(itertools.product(vals1, vals2))
    result = pd.DataFrame(combos, columns=[c1, c2])

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Cross join: {len(vals1)} × {len(vals2)} = {len(result)} combinations",
    )
