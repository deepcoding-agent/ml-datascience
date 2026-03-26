"""handle_apply_expr handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


def handle_apply_expr(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Apply a mathematical expression to create a new column.
    Expression uses column names as variables: 'price / area'."""
    expression = params.get("expression", "")
    new_name = params.get("new_name", "result")
    if not expression:
        return HandlerResult(success=False, error="Specify expression= parameter (e.g. 'price / area')")

    result = df.copy()
    try:
        result[new_name] = result.eval(expression)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created '{new_name}' = {expression}",
        )
    except Exception as e:
        return HandlerResult(success=False, error=f"Expression error: {e}")
