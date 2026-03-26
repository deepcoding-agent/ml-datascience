"""handle_log1p_transform handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_log1p_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Apply log(1+x) transform — safe for zero values, good for right-skewed data."""
    col = params.get("column")
    result = df.copy()
    if col and col in result.columns:
        result[f"{col}_log1p"] = np.log1p(result[col].clip(lower=0))
        summary = f"log1p-transformed '{col}' → '{col}_log1p'"
    else:
        transformed = []
        for c in result.select_dtypes(include="number").columns:
            if (result[c] >= 0).all():
                result[f"{c}_log1p"] = np.log1p(result[c])
                transformed.append(c)
        summary = f"log1p-transformed {len(transformed)} non-negative columns: {transformed}"
    return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)
