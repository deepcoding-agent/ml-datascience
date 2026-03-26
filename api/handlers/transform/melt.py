"""handle_melt handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_melt(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Unpivot/melt: wide → long format."""
    id_vars = params.get("id_vars", [])
    value_vars = params.get("value_vars", [])

    if not id_vars:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        id_vars = cat_cols[:2] if cat_cols else [df.columns[0]]
    id_vars = [c for c in id_vars if c in df.columns]

    if not value_vars:
        value_vars = [c for c in df.columns if c not in id_vars]
    value_vars = [c for c in value_vars if c in df.columns]

    result = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                     var_name="variable", value_name="value")
    return HandlerResult(success=True, result_df=result, output_type="generate",
                         summary=f"Melted: {len(value_vars)} columns → long format ({len(result):,} rows)")
