"""handle_sqrt_transform handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_sqrt_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Apply square root transform to numeric columns (good for moderate skew)."""
    col = params.get("column")
    result = df.copy()

    if col and col in result.columns:
        result[f"{col}_sqrt"] = np.sqrt(result[col].clip(lower=0))
        summary = f"Sqrt-transformed '{col}' → '{col}_sqrt'"
    else:
        transformed = []
        for c in result.select_dtypes(include="number").columns:
            if abs(result[c].skew()) > 0.5 and (result[c] >= 0).all():
                result[f"{c}_sqrt"] = np.sqrt(result[c])
                transformed.append(c)
        summary = f"Sqrt-transformed {len(transformed)} columns: {transformed}"

    return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)
