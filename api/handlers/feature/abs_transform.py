"""handle_abs_transform handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_abs_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Apply absolute value transform to numeric columns."""
    col = params.get("column")
    result = df.copy()
    if col and col in result.columns:
        result[f"{col}_abs"] = result[col].abs()
        summary = f"Absolute value of '{col}' → '{col}_abs'"
    else:
        transformed = []
        for c in result.select_dtypes(include="number").columns:
            if (result[c] < 0).any():
                result[f"{c}_abs"] = result[c].abs()
                transformed.append(c)
        summary = f"Absolute-value transformed {len(transformed)} columns with negatives"
    return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)
