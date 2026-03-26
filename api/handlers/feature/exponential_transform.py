"""handle_exponential_transform handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_exponential_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Apply exponential transform (e^x) to numeric columns."""
    col = params.get("column")
    result = df.copy()
    if col and col in result.columns:
        result[f"{col}_exp"] = np.exp(result[col].clip(upper=700))
        summary = f"Exponential transform of '{col}' → '{col}_exp'"
    else:
        transformed = []
        for c in result.select_dtypes(include="number").columns:
            if result[c].max() <= 700:
                result[f"{c}_exp"] = np.exp(result[c])
                transformed.append(c)
        summary = f"Exponential-transformed {len(transformed)} columns (max<=700)"
    return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)
