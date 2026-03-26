"""handle_flatten_columns handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_flatten_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Flatten multi-level column names into single-level snake_case names.
    Also standardizes all column names to lowercase snake_case."""
    result = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = ["_".join(str(c) for c in col if str(c) != "").strip("_")
                          for col in result.columns]

    import re as _re
    new_names = {}
    for c in result.columns:
        name = str(c).strip()
        name = _re.sub(r"[^\w\s]", "", name)
        name = _re.sub(r"\s+", "_", name)
        name = name.lower().strip("_")
        if not name:
            name = f"col_{list(result.columns).index(c)}"
        new_names[c] = name
    result = result.rename(columns=new_names)

    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Flattened {len(result.columns)} columns to snake_case",
    )
