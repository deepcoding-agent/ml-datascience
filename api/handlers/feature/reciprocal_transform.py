"""handle_reciprocal_transform handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_reciprocal_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Apply 1/x reciprocal transform (zeros become NaN)."""
    col = params.get("column")
    result = df.copy()
    if col and col in result.columns:
        safe = result[col].replace(0, np.nan)
        result[f"{col}_reciprocal"] = (1.0 / safe).round(6)
        summary = f"Reciprocal of '{col}' → '{col}_reciprocal'"
    else:
        transformed = []
        for c in result.select_dtypes(include="number").columns:
            safe = result[c].replace(0, np.nan)
            result[f"{c}_reciprocal"] = (1.0 / safe).round(6)
            transformed.append(c)
        summary = f"Reciprocal-transformed {len(transformed)} numeric columns"
    return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)
