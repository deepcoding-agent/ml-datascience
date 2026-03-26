"""handle_log_transform handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_log_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    col = params.get("column")
    result = df.copy()
    if col and col in result.columns:
        result[f"{col}_log"] = np.log1p(result[col])
        summary = f"Log-transformed '{col}' → '{col}_log'"
    else:
        skewed = []
        for c in result.select_dtypes(include="number").columns:
            if abs(result[c].skew()) > 1 and (result[c] >= 0).all():
                result[f"{c}_log"] = np.log1p(result[c])
                skewed.append(c)
        summary = f"Log-transformed {len(skewed)} skewed columns: {skewed}"
    return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)
