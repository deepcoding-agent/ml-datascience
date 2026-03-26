"""handle_feature_cross handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_feature_cross(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Cross two categorical columns to create A_x_B combination feature."""
    cols = params.get("columns", [])
    result = df.copy()
    cat_cols = result.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cols) >= 2:
        cols = cols[:2]
    elif len(cat_cols) >= 2:
        cols = cat_cols[:2]
    else:
        return HandlerResult(success=False, error="Need at least 2 categorical columns")
    name = f"{cols[0]}_x_{cols[1]}"
    result[name] = result[cols[0]].astype(str) + "_x_" + result[cols[1]].astype(str)
    return HandlerResult(
        success=True, result_df=result, output_type="generate",
        summary=f"Created feature cross '{name}' ({result[name].nunique()} unique combinations)",
    )
